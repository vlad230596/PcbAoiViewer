"""CLI: focus stacking of grids captured at several heights -> all-in-focus panorama, height map, 3D viewer.

Example:
    venv/Scripts/python.exe stackFocus.py "SocketTester2/captured_*" --ref 51.5
    venv/Scripts/python.exe stackFocus.py "SocketTester2/captured_*" --ref 51.5 --scale 0.5 --no-viewer
    venv/Scripts/python.exe -m http.server 8765 --directory out/stack
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

from FocusStack import FocusStack, findLayers
from GridStitcher import GridStitcher
import BuildViewer


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('pattern', help='glob of the height folders, the trailing number of a folder name is Z in mm')
    ap.add_argument('--ref', type=float, required=True, help='Z of the reference layer (board in focus)')
    ap.add_argument('--zmin', type=float, default=50.0, help='lowest Z used (layers below the board add nothing)')
    ap.add_argument('--zmax', type=float, default=1e9)
    ap.add_argument('--zstep', type=float, default=0.5, help='use only Z on this grid (0 = all folders)')
    ap.add_argument('--out', default='out/stack')
    ap.add_argument('--scale', type=float, default=1.0)
    ap.add_argument('--no-parallax-regions', action='store_true')
    ap.add_argument('--ortho-passes', type=int, default=1,
                    help='re-composite ortho-corrected with the measured heights (0 = plain geometry)')
    ap.add_argument('--parallax-z', type=float, nargs='*', default=[],
                    help='heights where parallax is measured too (tops of tall parts are sharp there)')
    ap.add_argument('--cell', type=float, default=8.0, help='depth cell, full-res px')
    ap.add_argument('--depth-sigma', type=float, default=0.3, help='fusion: Z width of the layer mix, mm')
    ap.add_argument('--conf-min', type=float, default=0.6, help='min log ratio of focus peak over the median layer')
    ap.add_argument('--no-layer-tiles', action='store_true', help='skip Deep Zoom tiles of every height')
    ap.add_argument('--no-marking', action='store_true',
                    help='skip the marking layer (best view and height per raised part, for reading the marking)')
    ap.add_argument('--no-viewer', action='store_true')
    ap.add_argument('--save-focus', action='store_true',
                    help='save focus_layers.npy (log focus per layer and cell) to tune the depth step offline')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    layers = [(z, d) for z, d in findLayers(args.pattern) if args.zmin <= z <= args.zmax and
              (args.zstep <= 0 or abs(z / args.zstep - round(z / args.zstep)) < 1e-6)]
    refFolder = dict(layers).get(args.ref)
    if refFolder is None:
        raise SystemExit(f'reference Z {args.ref} not among {[z for z, _ in layers]}')
    print(f'{len(layers)} layers: {[z for z, _ in layers]}')

    ref = GridStitcher(scale=args.scale)
    ref.loadFolder(refFolder)
    ref.loadImages()
    ref.computeFlatField()
    ref.calibrate()
    ref.measurePairs()
    ref.optimize()
    ref.compensateGains()
    ref.layoutCanvas()
    print(f'reference Z {args.ref}: canvas {ref.canvasSize}, pair residual median {ref.pairStats()["median"]:.2f} px')

    fs = FocusStack(ref, args.ref, cellPx=args.cell, depthSigmaMm=args.depth_sigma, confMin=args.conf_min)
    if not args.no_parallax_regions:
        fs.detectParallax([(z, d) for z, d in layers if z in args.parallax_z])
    fs.run(layers, args.ortho_passes)
    fs.depth()
    fused = fs.fuse()
    marking = None
    if not args.no_marking:
        marking, records = fs.markingLayer(fused)

    t = time.time()
    q = [cv2.IMWRITE_JPEG_QUALITY, 93]
    cv2.imwrite(os.path.join(args.out, 'focus.jpg'), fused, q)
    H, W = fused.shape[:2]
    f = 2000 / max(W, H)
    small = (int(W * f), int(H * f))
    cv2.imwrite(os.path.join(args.out, 'focus_preview.jpg'), cv2.resize(fused, small, interpolation=cv2.INTER_AREA), q)
    color, vmax = fs.heightColor()
    cv2.imwrite(os.path.join(args.out, 'height_color.png'), color)
    np.save(os.path.join(args.out, 'height.npy'), fs.height)
    np.save(os.path.join(args.out, 'focus_z.npy'), fs.zmap)
    np.save(os.path.join(args.out, 'board_z.npy'), fs.board)
    if args.save_focus:
        np.save(os.path.join(args.out, 'focus_layers.npy'), np.stack([l.focus for l in fs.used]))
    if ref.regions:
        cv2.imwrite(os.path.join(args.out, 'parallax_regions.jpg'), ref.drawParallaxOverlay(fused), q)
        cv2.imwrite(os.path.join(args.out, 'parallax_shift.png'), np.clip(ref.parallaxDiff * 8, 0, 255).astype(np.uint8))
    if marking is not None:
        cv2.imwrite(os.path.join(args.out, 'marking.jpg'), marking, q)
        with open(os.path.join(args.out, 'marking.json'), 'w') as fp:
            json.dump(records, fp, indent=1)
    cv2.imwrite(os.path.join(args.out, 'confidence.png'), np.clip(fs.conf * 60, 0, 255).astype(np.uint8))
    layerDir = os.path.join(args.out, 'layers')
    os.makedirs(layerDir, exist_ok=True)
    for l in fs.used:
        cv2.imwrite(os.path.join(layerDir, f'z{l.z:.1f}.jpg'), l.pano, q)
    fs.timings['save'] = time.time() - t

    if not args.no_viewer:
        t = time.time()
        BuildViewer.buildStack(fs, fused, args.out, vmax, layerTiles=not args.no_layer_tiles, marking=marking)
        fs.timings['viewer'] = time.time() - t

    fs.timings['total'] = time.time() - t0
    report = {'reference': ref.report(), 'stack': fs.report(), 'scale': args.scale}
    with open(os.path.join(args.out, 'report.json'), 'w') as fp:
        json.dump(report, fp, indent=2)
    print('depth:', fs.depthStats)
    print('timings:', report['stack']['timingsSec'])
    print(f'output: {os.path.abspath(args.out)}')


if __name__ == '__main__':
    main()
