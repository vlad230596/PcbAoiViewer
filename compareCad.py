"""CLI: compare the focus-stack height map with the board's STEP model (CadReference.py).

Example:
    venv/Scripts/python.exe compareCad.py out/stack --board boards/Antilatency.SocketTester.Main-r79 --align out/captured_51.5

Needs boards/<board>/model/ (stepToMesh.py, CAD venv) and the alignBoard.py fit of the reference panorama the stack
was built on (same canvas, --scale 1). Writes <stack>/cad/: cad_height.npy, error.npy, cad_owner.npy, *.bin for 3d.html,
parts.json/csv, summary.json, error.png, cad_height.png, heights_side_by_side.jpg, scatter.png, crops/; re-renders
<stack>/3d.html with the CAD modes.
"""
import argparse
import json
import os
import time

from CadReference import CadComparison
import BuildViewer

DEFAULT_CROPS = ('U1,U29,U18,U19,C73,C46_DC_5V,C101_DC_5V,L4_DC_5V,D25_DC_5V,TO_MAIN,UI1,UI2,J7,J29,SW1,X1,R4,C1,'
                 'C47_DC_5V,U10_DC_5V')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stack', help='stackFocus.py output folder')
    ap.add_argument('--board', required=True, help='boards/<board> with model/mesh.npz + parts.json')
    ap.add_argument('--align', default='out/captured_51.5', help='alignBoard.py output (board.json with align.M)')
    ap.add_argument('--markings', default=None, help='markings.json with presence (default: <align>/markings.json)')
    ap.add_argument('--out', default=None, help='default <stack>/cad')
    ap.add_argument('--no-shift', action='store_true', help='keep the alignBoard fit, no residual XY shift')
    ap.add_argument('--crops', default=DEFAULT_CROPS)
    ap.add_argument('--no-viewer', action='store_true', help='do not re-render 3d.html')
    a = ap.parse_args()

    t0 = time.time()
    out = a.out or os.path.join(a.stack, 'cad')
    markings = a.markings or os.path.join(a.align, 'markings.json')
    cmp = CadComparison(a.stack, a.board, a.align, markings)
    print('checks:', cmp.checks)
    print('STEP frame vs IPC-2581:', {k: v for k, v in cmp.frame.items()})
    s = cmp.run(fitShift=not a.no_shift)
    cmp.save(out, os.path.join(a.stack, 'focus.jpg'), [r for r in a.crops.split(',') if r])
    if not a.no_viewer and os.path.exists(os.path.join(a.stack, '3d.html')):
        BuildViewer.rebuild3d(a.stack)
    keys = ('shiftCells', 'shiftUm', 'fitCells', 'fitParts', 'fitLargeParts', 'cellsBoard', 'cellsParts', 'cellsPartsAfterFit',
            'validFraction', 'rangeTopMm')
    for k in keys:
        print(f'{k}: {s[k]}')
    print('by class:')
    for k, v in s['byClass'].items():
        print(f'  {k}: {v}')
    print('by package:')
    for k, v in s['byPackage'].items():
        print(f'  {k}: {v}')
    print('worst:', json.dumps(s['worstParts'][:8]))
    print('saturated:', [p['ref'] for p in s['saturatedParts']])
    print(f'output: {os.path.abspath(out)} in {time.time() - t0:.1f} s')


if __name__ == '__main__':
    main()
