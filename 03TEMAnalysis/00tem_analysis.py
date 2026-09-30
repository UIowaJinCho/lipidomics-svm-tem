from glob import glob
import os
import re
from itertools import combinations
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from svgelements import SVG, Path, Shape

"""
Measures autophagosomes and autolysosomes from Inkscape SVG annotations of TEM
micrographs and tests NR against R. Objects are aggregated to the cell and cells
to the biological replicate, so inference is at n=3 per state rather than per object.
"""

SVG_DIR = './annotations'
OUT_DIR = './results'

# 2 micron scale bar measured at 380 px
NM_PER_PX = 2000.0 / 380.0

# the micrograph is the top square of the tiff, the rest is the text banner
IMG_PX = 1888

# Points sampled along each bezier before the polygon is built; areas agree to
# within 0.1% across 150-1200 points.
SAMPLE_POINTS = 300

# stroke colours Inkscape writes into the style attribute
COLOR_CLASS = {
    '#ff0000': 'AP',
    '#f00': 'AP',
    '#00ff00': 'AL',
    '#0f0': 'AL'
}

# fixed before looking at any result, four endpoints at n=3
ENDPOINTS = ['ap_fraction', 'ap_density', 'median_ap_area', 'median_al_area']

# expected name NR_N1_cell3_f2.svg, with the original microscope naming as fallback
NAME_RE = re.compile(r'^(NR|R)_([A-Za-z0-9]+)_cell(\d+)_f(\d+)$', re.IGNORECASE)
LEGACY_RE = re.compile(r'^(TMZ\s+)?(N\d+)\s+Cell\s*(\d+)\s*-\s*(\d+)$', re.IGNORECASE)

os.makedirs(OUT_DIR, exist_ok=True)

field_um2 = (IMG_PX * NM_PER_PX / 1e3) ** 2

svg_files = sorted(glob(os.path.join(SVG_DIR, '*.svg')))
if not svg_files:
    raise SystemExit('no svg files found in %s' % SVG_DIR)

object_rows = []
skipped = []

for svg_file in svg_files:

    stem = os.path.splitext(os.path.basename(svg_file))[0]

    match = NAME_RE.match(stem)
    if match:
        status = match.group(1).upper()
        replicate = match.group(2).upper()
        cell = int(match.group(3))
        field = int(match.group(4))
    else:
        match = LEGACY_RE.match(stem.strip())
        if match:
            status = 'R' if match.group(1) else 'NR'
            replicate = match.group(2).upper()
            cell = int(match.group(3))
            field = int(match.group(4))
        else:
            skipped.append(os.path.basename(svg_file))
            continue


    document = SVG.parse(svg_file)

    # the document width in user units equals the full tiff width in pixels,
    # which converts svg units to nanometres
    units_per_px = float(document.width) / IMG_PX
    nm_per_unit = NM_PER_PX / units_per_px

    for element in document.elements():

        if not isinstance(element, (Path, Shape)):
            continue

        # class comes from the stroke colour, which also skips the embedded
        # raster, guides, and anything traced in an unexpected colour
        stroke = getattr(element, 'stroke', None)
        if stroke is None:
            continue
        try:
            hexcol = str(stroke.hexrgb).lower() if hasattr(stroke, 'hexrgb') else str(stroke).lower()
        except Exception:
            continue
        if hexcol not in COLOR_CLASS:
            continue

        # svgelements has already applied the group transforms, so the sampled
        # points are in document coordinates
        try:
            points = [element.point(t) for t in np.linspace(0.0, 1.0, SAMPLE_POINTS)]
            coords = [(float(p.x), float(p.y)) for p in points]
        except Exception:
            continue

        if len(coords) < 4:
            continue

        polygon = Polygon(coords)

        # buffer(0) repairs the occasional self-intersecting trace
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty or polygon.area <= 0:
            continue

        area_units = polygon.area
        perimeter_units = polygon.length

        object_rows.append({
            'object_id': getattr(element, 'id', None),
            'cls': COLOR_CLASS[hexcol],
            'status': status,
            'replicate': replicate,
            'cell': cell,
            'field': field,
            'file': os.path.basename(svg_file),
            'cell_uid': '%s_%s_cell%i' % (status, replicate, cell),
            'area_um2': area_units * (nm_per_unit ** 2) / 1e6,
            'perimeter_um': perimeter_units * nm_per_unit / 1e3,
            'circularity': (4 * np.pi * area_units / perimeter_units ** 2)
                           if perimeter_units > 0 else np.nan,
            'diameter_nm': 2 * np.sqrt(area_units / np.pi) * nm_per_unit,
            'centroid_x': polygon.centroid.x,
            'centroid_y': polygon.centroid.y
        })



objects = pd.DataFrame(object_rows)
if objects.empty:
    raise SystemExit('no annotated objects extracted, check COLOR_CLASS against the stroke colours')

# collapse objects to the cell, which is the natural unit here since each field
# is only one view of the same cell
cell_rows = []

for cell_uid, group in objects.groupby('cell_uid'):

    n_fields = group['field'].nunique()
    ap = group[group['cls'] == 'AP']
    al = group[group['cls'] == 'AL']
    n_ap = len(ap)
    n_al = len(al)
    imaged_um2 = n_fields * field_um2

    cell_rows.append({
        'cell_uid': cell_uid,
        'status': group['status'].iloc[0],
        'replicate': group['replicate'].iloc[0],
        'cell': group['cell'].iloc[0],
        'n_fields': n_fields,
        'imaged_um2': imaged_um2,
        'n_ap': n_ap,
        'n_al': n_al,
        # Dimensionless, so it needs no imaged-area denominator and is unaffected by
        # field overlap or by how many fields a cell happens to have. Primary endpoint.
        'ap_fraction': n_ap / (n_ap + n_al) if (n_ap + n_al) > 0 else np.nan,
        # overlapping fields make the imaged area an upper bound, supporting endpoint only
        'ap_density': 100 * n_ap / imaged_um2,
        # median rather than mean so one bad trace does not move the cell value
        'median_ap_area': ap['area_um2'].median() if n_ap else np.nan,
        'median_al_area': al['area_um2'].median() if n_al else np.nan
    })

cells = pd.DataFrame(cell_rows).sort_values(['status', 'replicate', 'cell'])

# collapse cells to the replicate, this is the unit the statistics run on
replicates = (cells.groupby(['status', 'replicate'])
                   .agg(**{endpoint: (endpoint, 'mean') for endpoint in ENDPOINTS},
                        n_cells=('cell_uid', 'count'),
                        n_ap=('n_ap', 'sum'),
                        n_al=('n_al', 'sum'))
                   .reset_index())

stat_rows = []

for endpoint in ENDPOINTS:

    present = replicates.dropna(subset=[endpoint])
    group_r = present.loc[present['status'] == 'R', endpoint].to_numpy(float)
    group_nr = present.loc[present['status'] == 'NR', endpoint].to_numpy(float)

    if len(group_r) < 2 or len(group_nr) < 2:
        raise SystemExit('not enough replicates for %s' % endpoint)

    observed = group_r.mean() - group_nr.mean()

    # with 3 vs 3 there are only C(6,3)=20 label assignments, so every one of
    # them is enumerated and the smallest two-sided p reachable is 0.10
    pooled = np.concatenate([group_r, group_nr])
    size = len(group_r)
    indices = range(len(pooled))
    differences = []

    for combo in combinations(indices, size):
        left = pooled[list(combo)]
        right = pooled[[i for i in indices if i not in combo]]
        differences.append(left.mean() - right.mean())

    differences = np.asarray(differences)
    p_value = float(np.mean(np.abs(differences) >= abs(observed) - 1e-12))

    # small-sample corrected standardised difference
    n1, n2 = len(group_r), len(group_nr)
    pooled_sd = np.sqrt(((n1 - 1) * group_r.std(ddof=1) ** 2 +
                         (n2 - 1) * group_nr.std(ddof=1) ** 2) / (n1 + n2 - 2))
    d = observed / pooled_sd if pooled_sd > 0 else np.nan
    correction = 1 - 3 / (4 * (n1 + n2) - 9)

    stat_rows.append({
        'endpoint': endpoint,
        'mean_R': group_r.mean(),
        'mean_NR': group_nr.mean(),
        'difference': observed,
        'hedges_g': d * correction if d == d else np.nan,
        'p_perm': p_value,
        'n_perms': len(differences)
    })

statistics = pd.DataFrame(stat_rows)

# benjamini-hochberg across the four pre-declared endpoints
p_values = statistics['p_perm'].to_numpy(float)
valid = ~np.isnan(p_values)
q_values = np.full_like(p_values, np.nan)

present_p = p_values[valid]
count = len(present_p)

if count > 0:
    order = np.argsort(present_p)
    ranked = present_p[order]
    scaled = ranked * count / (np.arange(count) + 1)
    scaled = np.minimum.accumulate(scaled[::-1])[::-1]
    adjusted = np.empty(count)
    adjusted[order] = np.clip(scaled, 0, 1)
    q_values[valid] = adjusted

statistics['q_fdr'] = q_values

for name, table in [('objects', objects), ('cells', cells),
                    ('replicates', replicates), ('statistics', statistics)]:
    out_path = os.path.join(OUT_DIR, 'tem_%s.csv' % name)
    table.to_csv(out_path, index=False)