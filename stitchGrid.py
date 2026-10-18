"""CLI: stitch a grid of "{x}_{y}.png" frames using machine coordinates as primary geometry.

Example:
    venv/Scripts/python.exe stitchGrid.py SocketTester2/captured_51.5
    venv/Scripts/python.exe stitchGrid.py SocketTester2/captured_51.5 --no-refine --out out/captured_51.5_norefine
    venv/Scripts/python.exe stitchGrid.py SocketTester2/captured_51.5 --parallax-regions --out out/captured_51.5_parallax
    venv/Scripts/python.exe stitchGrid.py SocketTester2/captured_51.5 --scale 0.25 --no-viewer
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

from GridStitcher import GridStitcher
import BuildViewer


def saveSeamCrops(stitcher, pano, folder, count, size=480):
    """Full-resolution crops centred on the seam between pairs of neighbouring frames."""
    if os.path.isdir(folder):
        for f in os.listdir(folder):
            if f.startswith('seam_'):
                os.remove(os.path.join(folder, f))
    os.makedirs(folder, exist_ok=True)
    centres = stitcher.centres()
    H, W = pano.shape[:2]
    pairs = [p for p in stitcher.pairs if p.valid and p.kind in ('x', 'y')]
    pairs.sort(key=lambda p: -np.linalg.norm(p.residual))
    # worst few + evenly spread others
    rest = pairs[count // 2:]
    chosen = pairs[:count // 2] + rest[::max(1, len(rest) // max(1, count - count // 2))]
    chosen = chosen[:count]
    s = int(size * stitcher.scale)
    names = []
    for p in chosen:
        c = ((centres[p.i] + centres[p.j]) / 2).astype(int)
        x0, y0 = max(0, c[0] - s // 2), max(0, c[1] - s // 2)
        crop = pano[y0:y0 + s, x0:x0 + s]
        name = f'seam_{p.kind}_{stitcher.frames[p.i].name}__{stitcher.frames[p.j].name}' \
               f'_res{np.linalg.norm(p.residual) / stitcher.scale:.1f}.png'
        cv2.imwrite(os.path.join(folder, name), crop)
        names.append(name)
    return names


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input', help='folder with x_y.png frames')
    ap.add_argument('--out', help='output folder (default out/<input folder name>)')
    ap.add_argument('--blend', choices=['multiband', 'voronoi', 'feather', 'hard'], default='multiband',
                    help='multiband: voronoi seam for details + wide cross-fade for brightness (default)')
    ap.add_argument('--band', type=float, default=12.0, help='voronoi seam transition half-width, full-res px')
    ap.add_argument('--low-band', type=float, default=150.0,
                    help='multiband: half-width of the brightness cross-fade, full-res px')
    ap.add_argument('--no-gain', action='store_true', help='disable per-frame exposure / white balance compensation')
    ap.add_argument('--parallax-regions', action='store_true',
                    help='tall components (regions where aligned frames still disagree) are taken whole from ONE frame, '
                         'the seam goes around them')
    ap.add_argument('--parallax-shift', type=float, default=10.0,
                    help='local shift between two aligned frames (full-res px) that marks parallax')
    ap.add_argument('--scale', type=float, default=1.0, help='downscale frames for a quick run, e.g. 0.25')
    ap.add_argument('--no-refine', action='store_true', help='place frames purely by calibrated machine coordinates')
    ap.add_argument('--no-flat-field', action='store_true', help='disable vignetting correction')
    ap.add_argument('--prior-sigma', type=float, default=25.0,
                    help='expected machine positioning error, full-res px (weight of the coordinate prior)')
    ap.add_argument('--max-deviation', type=float, default=30.0,
                    help='reject pair measurements deviating more than this from the machine prediction, full-res px '
                         '(off-board background lies at another depth and moves differently)')
    ap.add_argument('--png', action='store_true', help='also save lossless panorama.png (slow, large)')
    ap.add_argument('--no-viewer', action='store_true', help='skip Deep Zoom tiles + index.html')
    ap.add_argument('--seam-crops', type=int, default=8, help='number of full-res seam crops to save')
    args = ap.parse_args()

    name = os.path.basename(os.path.normpath(args.input))
    out = args.out or os.path.join('out', name)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()

    st = GridStitcher(scale=args.scale, flatField=not args.no_flat_field, refine=not args.no_refine,
                      priorSigmaPx=args.prior_sigma, maxDeviationPx=args.max_deviation)
    st.loadFolder(args.input)
    st.loadImages()
    st.computeFlatField()
    st.calibrate()
    st.measurePairs()   # with --no-refine the measurements are only used for diagnostics
    st.optimize()
    if not args.no_gain:
        st.compensateGains()
    st.lowBand = args.low_band
    st.layoutCanvas()
    print(f'canvas {st.canvasSize[0]} x {st.canvasSize[1]}')
    if args.parallax_regions:
        st.detectParallax(shiftPx=args.parallax_shift)
    pano = st.composite(blend=args.blend, band=args.band)

    t = time.time()
    cv2.imwrite(os.path.join(out, 'panorama.jpg'), pano, [cv2.IMWRITE_JPEG_QUALITY, 93])
    if args.png:
        cv2.imwrite(os.path.join(out, 'panorama.png'), pano, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    H, W = pano.shape[:2]
    f = 2000 / max(W, H)
    cv2.imwrite(os.path.join(out, 'preview.jpg'),
                cv2.resize(pano, (int(W * f), int(H * f)), interpolation=cv2.INTER_AREA),
                [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imwrite(os.path.join(out, 'debug_overlay.jpg'), st.drawDebugOverlay(pano), [cv2.IMWRITE_JPEG_QUALITY, 90])
    if args.parallax_regions:
        cv2.imwrite(os.path.join(out, 'parallax_regions.jpg'), st.drawParallaxOverlay(pano), [cv2.IMWRITE_JPEG_QUALITY, 90])
        d = st.parallaxDiff
        cv2.imwrite(os.path.join(out, 'parallax_shift.png'), np.clip(d * 8, 0, 255).astype(np.uint8))  # 8 grey per px
    if args.seam_crops > 0:
        saveSeamCrops(st, pano, os.path.join(out, 'seams'), args.seam_crops)
    st.timings['save'] = time.time() - t

    if not args.no_viewer:
        t = time.time()
        BuildViewer.build(st, pano, out)
        st.timings['viewer'] = time.time() - t

    st.timings['total'] = time.time() - t0
    report = st.report()
    report['input'] = os.path.abspath(args.input)
    report['blend'] = args.blend
    report['panoramaSizeFullResPx'] = [int(W / args.scale), int(H / args.scale)]
    with open(os.path.join(out, 'report.json'), 'w') as fp:
        json.dump(report, fp, indent=2)
    with open(os.path.join(out, 'pairs.json'), 'w') as fp:
        json.dump([dict(p, a=st.frames[p['a']].name, b=st.frames[p['b']].name) for p in st.pairList()], fp, indent=1)

    cal = report['calibration']
    print(f"px/mm X {cal['pxPerMmX']:.2f}, Y {cal['pxPerMmY']:.2f}; machine X axis at {cal['machineXAxisAngleDeg']:.2f} deg, "
          f"Y axis at {cal['machineYAxisAngleDeg']:.2f} deg on the image")
    print('pair residuals (px):', {k: round(v, 2) for k, v in report['pairResidualsPx_validPairs'].items()})
    print(f"pairs used {report['pairs']['used']}/{report['pairs']['candidates']}, rejected {len(report['pairs']['rejected'])}")
    print('timings:', report['timingsSec'])
    print(f'output: {os.path.abspath(out)}')


if __name__ == '__main__':
    main()
