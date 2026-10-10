"""The all-scroll review gate must pass before requesting or initializing CUDA."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def require_review_targets():
    path=ROOT/'segmentation-review/inference-gate.json'
    state=json.loads(path.read_text()) if path.exists() else {}
    if state.get('status')!='ready':
        raise RuntimeError('Ink inference is paused pending the fixed, saved user-approved selection.')
    overnight=state.get('selection_policy')=='user_authorized_overnight_approved_regions_20260907'
    if overnight:
        if state.get('authorization')!='Yes—run the approved regions tonight':
            raise RuntimeError('The overnight target exception requires the explicit user authorization')
        if not state.get('final_review_complete'):
            raise RuntimeError('Tonight\'s final review selection is not fixed yet')
    else:
        for scroll in ['PHerc0841','PHerc0814']:
            if state.get('approved_distinct_lower_cm2',{}).get(scroll,0)<200:
                raise RuntimeError('Approved distinct surface target has not been established: '+scroll)
        if not state.get('9b_selection_complete'):
            raise RuntimeError('Best-available 9B review selection is not complete')
    if not state.get('approval_revisions'):
        raise RuntimeError('Gate has no saved user review revisions')
    for ident,expected in state['approval_revisions'].items():
        folder=ROOT/'segmentation-review'/ident
        files=sorted((folder/'decisions').glob('r*.json'))
        if not files:raise RuntimeError('Missing saved review: '+ident)
        current=json.loads(files[-1].read_text())
        meta=json.loads((folder/'review.json').read_text())
        for key in ['revision','mask_sha256','geometry_fingerprint']:
            if current.get(key)!=expected.get(key):raise RuntimeError('Review changed since area accounting: '+ident)
        if meta['geometry_fingerprint']!=current['geometry_fingerprint']:
            raise RuntimeError('Geometry changed since review: '+ident)
    return state
