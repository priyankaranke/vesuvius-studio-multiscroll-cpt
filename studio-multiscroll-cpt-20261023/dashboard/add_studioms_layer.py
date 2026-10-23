"""Add the 'studioms' layer (Studio after multi-scroll continued pretraining) to a manual-revision family and its 8798 viewer.

Patches, with *.before-studioms-20261023 backups: the family's register3d.py / make_jpegs.py / build_overlays.py /
update_viewer_history.py model lists, and the viewer's viewer.html (button key 8, preview files, label) and versions.json
(+ the inline catalog copy in viewer.html). Idempotent.
usage: add_studioms_layer.py <family dir> <viewer dir>"""
import json, re, shutil, sys
from pathlib import Path

family, viewer = Path(sys.argv[1]), Path(sys.argv[2])
NAME = 'studioms'; LABEL = 'Studio · multi-scroll CPT (23 Oct)'
TITLE = 'Studio epoch 9 continued on 12 other scroll groups (Expanded9 corpus + Reader v2/v3 labels) with 814/841/9B replay every batch, 23 Oct 2026'


def patch(path, pairs):
    s = path.read_text(); orig = s
    for old, new, guard in pairs:
        if guard in s: continue
        assert old in s, (path, old[:60]); s = s.replace(old, new, 1)
    if s != orig:
        backup = path.with_name(path.name + '.before-studioms-20261023')
        if not backup.exists(): shutil.copy2(path, backup)
        path.write_text(s); print('PATCHED', path)


patch(family/'register3d.py', [("for m in ('studio', 'thin15', 'readerv3', 'expanded9', 'e9all'):", "for m in ('studio', 'thin15', 'readerv3', 'expanded9', 'e9all', 'studioms'):", "'e9all', 'studioms'")])
patch(family/'make_jpegs.py', [('"expanded9", "e9all"]', '"expanded9", "e9all", "studioms"]', '"studioms"')])
patch(family/'build_overlays.py', [("'e9all': [0.0, 1.0]}", "'e9all': [0.0, 1.0], 'studioms': [0.0, 1.0]}", "'studioms'")])
patch(family/'update_viewer_history.py', [("+(['e9all'] if (revision/'e9all.npy').exists() else [])", "+(['e9all'] if (revision/'e9all.npy').exists() else [])+(['studioms'] if (revision/'studioms.npy').exists() else [])", "'studioms'")])

html = viewer/'viewer.html'
patch(html, [
    ('<button data-layer="ct"', f'<button data-layer="{NAME}" aria-pressed="false" title="{TITLE}"><kbd>8</kbd>Studio · multi-scroll</button><button data-layer="ct"', f'data-layer="{NAME}"'),
    ("'e9all-probability':'e9all.preview.jpg',", f"'e9all-probability':'e9all.preview.jpg','{NAME}':'{NAME}-turbo.preview.jpg','{NAME}-probability':'{NAME}.preview.jpg',", f"'{NAME}-probability'"),
    ("e9all:'Expanded9 · 211 FT (all labels)'}", f"e9all:'Expanded9 · 211 FT (all labels)',{NAME}:'{LABEL}'}}", f"{NAME}:'"),
    ("'7':'e9all'}", f"'7':'e9all','8':'{NAME}'}}", f"'8':'{NAME}'"),
])
versions = json.loads((viewer/'versions.json').read_text())
for r in versions['revisions']:
    if (viewer/r['base']/f'{NAME}.npy').exists() and NAME not in r['models']: r['models'].append(NAME)
(viewer/'versions.json').write_text(json.dumps(versions, indent=2) + '\n')
s = html.read_text()
s2 = re.sub(r'(<script type="application/json" id="surface-versions">)(.*?)(</script>)', lambda m: m.group(1) + '\n' + json.dumps(versions, indent=2) + '\n' + m.group(3), s, count=1, flags=re.S)
assert s2 != s or json.dumps(versions, indent=2) in s, 'inline catalog not found'
html.write_text(s2); print('VIEWER_CATALOG', [r['models'] for r in versions['revisions']])
