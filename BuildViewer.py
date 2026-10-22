"""Deep Zoom (DZI) tile pyramid + static OpenSeadragon viewer for a stitched panorama.

The output folder works when served with:  python -m http.server 8000 --directory out/<name>
(open http://localhost:8000/). Tiles are written with OpenCV, no pyvips needed.
"""
import json
import math
import os
import shutil
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

OSD_VERSION = '4.1.0'
OSD_URL = f'https://cdnjs.cloudflare.com/ajax/libs/openseadragon/{OSD_VERSION}/openseadragon.min.js'
OSD_IMAGES = f'https://cdnjs.cloudflare.com/ajax/libs/openseadragon/{OSD_VERSION}/images/'
THREE_VERSION = '0.160.0'


def writeDzi(image, outDir, name='tiles', tileSize=256, overlap=1, quality=88, threads=8):
    """Writes <outDir>/<name>.dzi and <outDir>/<name>_files/<level>/<col>_<row>.jpg."""
    filesDir = os.path.join(outDir, f'{name}_files')
    if os.path.isdir(filesDir):
        shutil.rmtree(filesDir)
    H, W = image.shape[:2]
    maxLevel = int(math.ceil(math.log2(max(W, H))))
    level = image
    jobs = []
    params = [cv2.IMWRITE_JPEG_QUALITY, quality]

    def writeTile(args):
        path, tile = args
        cv2.imwrite(path, tile, params)

    with ThreadPoolExecutor(threads) as pool:
        for lv in range(maxLevel, -1, -1):
            h, w = level.shape[:2]
            d = os.path.join(filesDir, str(lv))
            os.makedirs(d, exist_ok=True)
            tiles = []
            for row in range(int(math.ceil(h / tileSize))):
                for col in range(int(math.ceil(w / tileSize))):
                    x0 = max(0, col * tileSize - overlap)
                    y0 = max(0, row * tileSize - overlap)
                    x1 = min(w, (col + 1) * tileSize + overlap)
                    y1 = min(h, (row + 1) * tileSize + overlap)
                    tiles.append((os.path.join(d, f'{col}_{row}.jpg'), level[y0:y1, x0:x1]))
            list(pool.map(writeTile, tiles))
            jobs.append(len(tiles))
            if lv > 0:
                level = cv2.resize(level, (max(1, (w + 1) // 2), max(1, (h + 1) // 2)), interpolation=cv2.INTER_AREA)
    with open(os.path.join(outDir, f'{name}.dzi'), 'w') as fp:
        fp.write('<?xml version="1.0" encoding="UTF-8"?>\n'
                 f'<Image xmlns="http://schemas.microsoft.com/deepzoom/2008" TileSize="{tileSize}" '
                 f'Overlap="{overlap}" Format="jpg"><Size Width="{W}" Height="{H}"/></Image>\n')
    return sum(jobs)


def seamImage(stitcher, factor=4):
    label = stitcher.labelMap(factor)
    edge = np.zeros(label.shape, bool)
    edge[:, 1:] |= label[:, 1:] != label[:, :-1]
    edge[1:, :] |= label[1:, :] != label[:-1, :]
    rgba = np.zeros(label.shape + (4,), np.uint8)
    rgba[edge] = (0, 230, 255, 255)   # BGR yellow
    return rgba


def writeFocusZ(outDir, sigmaCells=1.0):
    """focusz.bin for the page's floor fusion: depth of best focus per cell (focus_z.npy, Z mm), blurred as in
    FocusStack.fuse, int16 um, row-major cells."""
    z = np.load(os.path.join(outDir, 'focus_z.npy')).astype(np.float32)
    z = cv2.GaussianBlur(z, (0, 0), sigmaCells)
    np.clip(np.round(z * 1000), -32000, 32000).astype('<i2').tofile(os.path.join(outDir, 'focusz.bin'))


def rebuildHtml(outDir, board=None):
    """Re-renders index.html with the current TEMPLATE and the DATA of the page already there (no re-stitching).
    For viewer changes; a focus-stack page keeps its stack data too. board: a panorama folder of the same reference
    stitch whose board.json / markings / marking views the page shows (kept in DATA.boardBase for later rebuilds;
    serve a common parent folder). A stack page also gets focusz.bin for fusing any set of heights."""
    path = os.path.join(outDir, 'index.html')
    with open(path, encoding='utf-8') as fp:
        html = fp.read()
    start = html.index('const DATA = ') + len('const DATA = ')
    end = html.index(';\nconst s = DATA.scale;', start)
    data = json.loads(html[start:end])
    if board:
        data['boardBase'] = os.path.relpath(board, outDir).replace(os.sep, '/') + '/'
    if data.get('stack'):
        data['stack'].setdefault('depthSigma', 0.3)          # FocusStack depthSigmaMm default
        if os.path.exists(os.path.join(outDir, 'focus_z.npy')):
            writeFocusZ(outDir)
    out = TEMPLATE.replace('__OSD_URL__', OSD_URL).replace('__OSD_IMAGES__', OSD_IMAGES) \
        .replace('__DATA__', json.dumps(data))
    with open(path, 'w', encoding='utf-8') as fp:
        fp.write(out)


def rebuild3d(outDir):
    """Re-renders 3d.html with the current TEMPLATE_3D and the data of the page already there (no re-stacking).
    The page picks up cad/viewer.json (compareCad.py) by itself when it is there."""
    path = os.path.join(outDir, '3d.html')
    with open(path, encoding='utf-8') as fp:
        html = fp.read()
    start = html.index('const D = ') + len('const D = ')
    end = html.index(';\nconst status = ', start)
    with open(path, 'w', encoding='utf-8') as fp:
        fp.write(TEMPLATE_3D.replace('__DATA__', html[start:end]).replace('__THREE__', THREE_VERSION))


def build(stitcher, pano, outDir, stack=None):
    """stack: focus-stack extras for the page (see buildStack), None for a single-height panorama."""
    H, W = pano.shape[:2]
    nTiles = writeDzi(pano, outDir)
    factor = 4
    cv2.imwrite(os.path.join(outDir, 'seams.png'), seamImage(stitcher, factor))

    framesDir = os.path.join(outDir, 'frames')
    os.makedirs(framesDir, exist_ok=True)
    fw, fh = stitcher.frameSize
    thumbW = 640
    sourcesDir = os.path.join(outDir, 'sources')
    os.makedirs(sourcesDir, exist_ok=True)

    def writeFrame(f):
        t = cv2.resize(f.image, (thumbW, int(fh * thumbW / fw)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(framesDir, f'{f.name}.jpg'), t, [cv2.IMWRITE_JPEG_QUALITY, 85])
        # untouched original (no flat field / gain), full resolution, for the click inspector
        cv2.imwrite(os.path.join(sourcesDir, f'{f.name}.jpg'), cv2.imread(f.path, cv2.IMREAD_COLOR),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(writeFrame, stitcher.frames))

    data = {
        'width': W, 'height': H, 'frameW': fw, 'frameH': fh, 'scale': stitcher.scale,
        'seamsFactor': factor,
        'frames': [{'name': f.name, 'x': f.x, 'y': f.y, 'path': os.path.abspath(f.path),
                    'px': round(float(f.position[0]), 2), 'py': round(float(f.position[1]), 2)}
                   for f in stitcher.frames],
        'pairs': stitcher.pairList(),
        'calibration': stitcher.calibrationSummary(),
        'stats': stitcher.pairStats(),
        # parallax regions taken whole from one frame (empty unless enabled; not used by an ortho-corrected stack)
        'regions': stack.get('regions', stitcher.regions) if stack else stitcher.regions,
        'stack': stack,
    }
    html = TEMPLATE.replace('__OSD_URL__', OSD_URL).replace('__OSD_IMAGES__', OSD_IMAGES) \
        .replace('__DATA__', json.dumps(data))
    with open(os.path.join(outDir, 'index.html'), 'w', encoding='utf-8') as fp:
        fp.write(html)
    return nTiles


def buildStack(fs, fused, outDir, heightMax, layerTiles=True, meshVertices=1.5e6, marking=None):
    """Viewer for a FocusStack: all-in-focus panorama as the main image, every height as a switchable layer,
    the marking layer (FocusStack.markingLayer) if given, height map overlay + readout, and 3d.html (three.js
    relief textured with the all-in-focus panorama)."""
    ref = fs.ref
    layers = []
    for l in fs.used:
        name = f'z{l.z:.1f}'
        if layerTiles:
            writeDzi(l.pano, outDir, name=name)
        hasTiles = layerTiles or os.path.exists(os.path.join(outDir, f'{name}.dzi'))
        layers.append({'z': l.z, 'tiles': f'{name}.dzi' if hasTiles else None,
                       'aboveBoardMm': round(l.z - fs.depthStats['boardZ_mm'], 2)})
    # heights in um, int16, row-major cells (cell = fs.factor work px)
    hs, ws = fs.height.shape
    np.clip(np.round(fs.height * 1000), -32000, 32000).astype('<i2').tofile(os.path.join(outDir, 'height.bin'))
    mesh = fs.meshHeights()
    np.where(np.isfinite(mesh), np.clip(np.round(np.nan_to_num(mesh) * 1000), -32000, 32000), -32768)         .astype('<i2').tofile(os.path.join(outDir, 'mesh.bin'))   # -32768 = not covered by any frame
    (fs.valid | fs.top).astype(np.uint8).tofile(os.path.join(outDir, 'measured.bin'))   # 1 = measured, 0 = filled
    color, _ = fs.heightColor(heightMax)
    cv2.imwrite(os.path.join(outDir, 'height_color.png'), color)
    H, W = fused.shape[:2]
    for side, name in ((8192, 'focus_tex.jpg'), (4096, 'focus_tex_4k.jpg')):
        f = min(1.0, side / max(W, H))
        cv2.imwrite(os.path.join(outDir, name), cv2.resize(fused, (int(W * f), int(H * f)), interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
    cal = ref.calibrationSummary()
    stack = {
        'layers': layers,
        'cells': [ws, hs], 'cell': fs.factor, 'heightMax': heightMax,
        'boardZ': fs.depthStats['boardZ_mm'], 'cameraDistanceMm': round(fs.distance, 2),
        'orthoPasses': fs.orthoPasses,
        'mmPerPx': cal['umPerPx'] / 1000.0 / ref.scale,      # per work px
        'meshStep': max(1, int(math.ceil(math.sqrt(ws * hs / meshVertices)))),
    }
    if marking is not None:
        writeDzi(marking, outDir, name='marking')
        stack['marking'] = {'tiles': 'marking.dzi', 'parts': fs.markingRecords}
    if fs.orthoPasses:
        stack['regions'] = []
    build(ref, fused, outDir, stack)
    data = {'width': W, 'height': H, **stack}
    with open(os.path.join(outDir, '3d.html'), 'w', encoding='utf-8') as fp:
        fp.write(TEMPLATE_3D.replace('__DATA__', json.dumps(data)).replace('__THREE__', THREE_VERSION))


TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PCB Panorama</title>
<script src="__OSD_URL__"></script>
<style>
  html, body { margin: 0; height: 100%; background: #111; color: #ddd; font: 13px system-ui, sans-serif; }
  #viewer { position: absolute; inset: 0; }
  #overlay { position: absolute; inset: 0; pointer-events: none; }
  #panel { position: absolute; top: 8px; left: 8px; background: rgba(20,20,20,.85); padding: 8px 10px;
           border-radius: 6px; z-index: 10; max-width: 340px; }
  #panel label { display: block; margin: 2px 0; cursor: pointer; }
  #info { position: absolute; top: 8px; right: 8px; bottom: 8px; background: rgba(20,20,20,.92); padding: 8px 10px;
          border-radius: 6px; z-index: 10; width: 652px; max-width: calc(100vw - 36px); display: none;
          overflow-y: auto; box-sizing: border-box; }
  #info .head { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
  #info button { background: #333; color: #ddd; border: 1px solid #555; border-radius: 4px; cursor: pointer; }
  #info button.on { background: #0a6; border-color: #0c8; color: #fff; }
  #cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 8px; margin-top: 8px; }
  .card { background: #1c1c1c; border: 1px solid #333; border-radius: 5px; padding: 5px; }
  .card.used { border-color: #0c8; }
  .card .crop { position: relative; height: 240px; background: #000; overflow: hidden; cursor: zoom-in;
                border-radius: 3px; }
  .card .crop img { position: absolute; max-width: none; image-rendering: auto; }
  .card .crop::before, .card .crop::after { content: ''; position: absolute; z-index: 1; background: rgba(255,0,255,.8); }
  .card .crop::before { left: 50%; top: calc(50% - 12px); width: 1px; height: 24px; }
  .card .crop::after { top: 50%; left: calc(50% - 12px); height: 1px; width: 24px; }
  .card .whole { position: relative; margin-top: 4px; }
  .card .whole img { width: 100%; display: block; }
  .card .whole i { position: absolute; width: 8px; height: 8px; margin: -5px 0 0 -5px; border: 1px solid #f0f;
                   border-radius: 50%; }
  .tag { font-size: 11px; padding: 0 4px; border-radius: 3px; background: #0a6; color: #fff; margin-left: 4px; }
  .legend span { display: inline-block; width: 10px; height: 10px; margin: 0 3px 0 8px; vertical-align: middle; }
  small { color: #999; }
  #cadPanel select, #cadPanel input[type=text] { background: #222; color: #ddd; border: 1px solid #444; border-radius: 4px; }
  #cadLegend span.k { display: inline-flex; align-items: center; gap: 4px; margin-right: 8px; cursor: pointer; font-size: 12px; }
  #cadLegend span.k.off { opacity: .35; }
  #cadLegend i { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
  .part .kv { display: grid; grid-template-columns: 100px 1fr; gap: 2px 10px; margin-top: 6px; }
  .part .kv b { font-weight: 500; color: #999; }
  .part code { background: #2a2f35; padding: 1px 6px; border-radius: 4px; margin-right: 6px; }
  .part .md { border-top: 1px solid #333; margin-top: 6px; padding-top: 4px; overflow-x: auto; }
  .part .md table { border-collapse: collapse; font-size: 12px; margin: 4px 0; }
  .part .md td, .part .md th { border: 1px solid #444; padding: 2px 5px; vertical-align: top; }
  .part .md img { max-width: 100%; background: #fff; border-radius: 4px; }
  .part .md h3, .part .md h4, .part .md h5, .part .md h6 { font-size: 13px; margin: 8px 0 4px; }
  .part details { margin: 5px 0; }
  .part summary { cursor: pointer; color: #6cf; }
  .part a { color: #6cf; }
  .fit { display: inline-block; padding: 0 7px; border-radius: 10px; border: 1px solid currentColor; font-size: 12px; }
  #cadPanel button { background: #333; color: #ddd; border: 1px solid #555; border-radius: 4px; cursor: pointer; }
  #hud { position: absolute; left: 8px; bottom: 8px; z-index: 12; background: rgba(10,10,10,.9); border: 1px solid #333;
         border-radius: 6px; padding: 6px 9px; font: 12px/1.4 monospace; white-space: pre; display: none; }
  #hud button { background: #333; color: #ddd; border: 1px solid #555; border-radius: 4px; cursor: pointer; font: 12px system-ui; }
  #list { position: fixed; inset: 0; z-index: 30; background: #121212; overflow-y: auto; display: none; }
  #list .lhead { position: sticky; top: 0; z-index: 1; background: #181818; border-bottom: 1px solid #333; padding: 8px 14px;
                 display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
  #list .lhead select, #list .lhead input[type=text] { background: #222; color: #ddd; border: 1px solid #444; border-radius: 4px; padding: 3px 6px; }
  #list .lhead button { background: #333; color: #ddd; border: 1px solid #555; border-radius: 4px; cursor: pointer; padding: 3px 9px; }
  #lBody { padding: 4px 14px 30px; }
  #lBody h2 { font-size: 15px; margin: 18px 0 4px; border-bottom: 1px solid #333; padding-bottom: 3px; }
  #lBody h2 small, .lline small { color: #888; font-weight: normal; }
  .lline { margin: 10px 0 4px; }
  .lline code { background: #2a2f35; padding: 1px 6px; border-radius: 4px; margin-left: 4px; color: #ffd36b; }
  .lcards { display: flex; flex-wrap: wrap; gap: 6px; }
  .lcard { background: #1c1c1c; border: 1px solid #333; border-radius: 5px; padding: 3px; cursor: pointer; text-align: center; }
  .lcard:hover { border-color: #6cf; }
  .lcard img { display: block; height: var(--h, 96px); max-width: calc(var(--h, 96px) * 4); object-fit: contain; background: #000; }
  .lcard .noimg { height: var(--h, 96px); width: var(--h, 96px); display: flex; align-items: center; justify-content: center;
                  color: #666; font-size: 11px; background: #0b0b0b; }
  .lcard div.r { font: 12px monospace; margin-top: 2px; }
  .lcard .mk { font: 12px/1.25 monospace; color: #ffd36b; margin-top: 2px; min-height: 15px; }
  .lcard .ck { font-size: 11px; margin-top: 1px; max-width: calc(var(--h, 96px) * 3); }
  .lcard .oc { font: 11px monospace; margin-top: 1px; }
  .zchips { margin: 2px 0 4px; }
  .zchips button.zc { background: #222; color: #aaa; border: 1px solid #444; border-radius: 3px; padding: 1px 5px; margin: 0 2px 2px 0; font-size: 11px; cursor: pointer; }
  .zchips button.zc.on { background: #1d4b66; color: #fff; border-color: #6cf; }
  .imgs { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 6px; }
  .imgs figure { margin: 0; }
  .imgs img { max-height: 240px; max-width: 300px; border-radius: 4px; display: block; background: #000; }
  .imgs figcaption { font-size: 11px; color: #999; margin-top: 2px; max-width: 300px; }
  .mkread { font: 13px/1.3 monospace; color: #ffd36b; background: #1c1c1c; padding: 3px 6px; border-radius: 4px; display: inline-block; }
</style>
</head>
<body>
<div id="viewer"></div>
<canvas id="overlay"></canvas>
<div id="panel">
  <b>PCB panorama</b> <small id="size"></small><br>
  <label><input type="checkbox" id="cbFrames"> frame outlines + x_y labels</label>
  <label><input type="checkbox" id="cbSeams"> voronoi seams</label>
  <label><input type="checkbox" id="cbPairs"> pair residuals</label>
  <label id="lblRegions"><input type="checkbox" id="cbRegions"> parallax regions (one frame each)</label>
  <div class="legend" id="legend" style="display:none">
    <span style="background:#2d2"></span>&lt;1px<span style="background:#dd2"></span>&lt;2.5<span style="background:#e22"></span>&ge;2.5
    <span style="background:#e2e"></span>rejected<span style="background:#888"></span>unused
  </div>
  <small>click: all source frames that contain the point; double-click: zoom</small><br>
  <small id="stats"></small>
  <div id="stackPanel" style="display:none">
    <hr style="border-color:#333">
    <b>Focus stack</b> <a href="3d.html" style="color:#6cf;margin-left:8px">3D view &rarr;</a><br>
    <label><input type="radio" name="layer" value="-1" checked> all-in-focus</label>
    <div id="markingRow" style="display:none;align-items:center;gap:10px">
      <label><input type="radio" name="layer" value="m"> marking: best view + height per part</label>
      <label><input type="checkbox" id="cbParts" checked> outlines</label>
    </div>
    <div id="floorRow">
      floors <span id="floorChips" class="zchips"></span><br>
      <small>click: this floor only · ctrl+click: add / remove · all = all-in-focus; several = focus fusion of them only</small>
      <label id="lblTops" style="display:none"><input type="checkbox" id="cbTops"> part tops from source views
        <small>(mosaic of the frames, the one the VLM read on top; sharpest of the picked floors)</small></label>
    </div>
    <div style="display:flex;align-items:center;gap:6px">
      <label style="white-space:nowrap"><input type="radio" name="layer" value="z"> height</label>
      <input type="range" id="zSlider" min="0" max="0" step="1" value="0" style="flex:1">
      <span id="zLabel" style="min-width:110px"></span>
    </div>
    <label><input type="checkbox" id="cbHeight"> height map, opacity
      <input type="range" id="hOpacity" min="0" max="100" value="60" style="width:90px;vertical-align:middle"></label>
    <label>colour range 0 &hellip; <input type="range" id="hRange" min="0.3" max="5" step="0.1" value="1.5"
      style="width:110px;vertical-align:middle"> <span id="hRangeV"></span> mm</label>
    <div id="hLegend" style="display:none;margin:2px 0 4px"></div>
    <small>[ / ] step through heights, f = all-in-focus, m = marking</small><br>
    <span id="hCursor"></span>
  </div>
  <div id="cadPanel" style="display:none">
    <hr style="border-color:#333">
    <b>Board (CAD)</b> <small id="cadInfo"></small><br>
    <label style="display:inline-block"><input type="checkbox" id="cbCad" checked> components</label>
    <label style="display:inline-block;margin-left:8px"><input type="checkbox" id="cbPads"> pads</label>
    <label style="display:inline-block;margin-left:8px" id="lblFind" hidden><input type="checkbox" id="cbFindings" checked> findings <small id="findCount"></small></label>
    <select id="cadMode" style="margin-left:8px"><option value="fit">colour: fit</option><option value="group">colour: group</option></select>
    <div id="cadLegend" style="margin:4px 0"></div>
    <input type="text" id="cadFind" placeholder="R12, U1, 10k&hellip;" size="16">
    <button id="btnList">parts list</button>
    <small>click a part: its card</small>
  </div>
  <hr style="border-color:#333">
  <label><input type="checkbox" id="cbHud"> diagnostics (fps, draw, tiles) <small>&mdash; key D</small></label>
</div>
<div id="info"></div>
<div id="hud"></div>
<div id="list">
  <div class="lhead">
    <b>Parts</b> <small id="lCount"></small>
    <select id="lGroup"></select>
    <select id="lSide"><option value="">both sides</option><option value="top">top</option><option value="bottom">bottom</option></select>
    <select id="lFit"><option value="">any fit</option><option value="fitted">fitted</option><option value="nc">NC</option>
      <option value="optional">optional</option><option value="none">not in BOM</option></select>
    <select id="lCheck"><option value="">any check</option><option value="problems">problems</option>
      <option value="ok">ok</option><option value="unread">unread</option><option value="text">marking read</option>
      <option value="ocrdiff">OCR differs</option><option value="ocrsame">OCR agrees</option>
      <option value="codediff">code differs</option><option value="vlmdiff">VLM differs</option><option value="viewsdiff">views differ / new</option></select>
    <input type="text" id="lFind" placeholder="ref, name, value, marking" size="16">
    <label>preview <input type="range" id="lSize" min="48" max="260" value="110" style="vertical-align:middle"></label>
    <span style="flex:1"></span>
    <button id="lClose">close &times;</button>
  </div>
  <div id="lBody"></div>
</div>
<script>
const DATA = __DATA__;
const s = DATA.scale;
document.getElementById('size').textContent = DATA.width + ' x ' + DATA.height + ' px';
if (DATA.stats && DATA.stats.count)
  document.getElementById('stats').textContent =
    'pair residual median ' + DATA.stats.median.toFixed(2) + ' px, p90 ' + DATA.stats.p90.toFixed(2) +
    ', max ' + DATA.stats.max.toFixed(2) + ' (' + DATA.stats.count + ' pairs)';

const viewer = OpenSeadragon({
  id: 'viewer', prefixUrl: '__OSD_IMAGES__', tileSources: 'tiles.dzi',
  maxZoomPixelRatio: 4, showNavigator: true, navigatorPosition: 'BOTTOM_RIGHT',
  gestureSettingsMouse: { clickToZoom: false, dblClickToZoom: true },
  imageLoaderLimit: 8, immediateRender: false, animationTime: 0.6,
});

let seamsItem = null;
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const cb = id => document.getElementById(id);

function main() { return viewer.world.getItemAt(0); }
function toScreen(x, y) {
  const p = main().imageToViewerElementCoordinates(new OpenSeadragon.Point(x, y));
  return [p.x, p.y];
}
function centre(f) { return [f.px + (DATA.frameW - 1) / 2, f.py + (DATA.frameH - 1) / 2]; }
function pairColor(p) {
  if (p.valid) return p.residualPx < 1 ? '#2d2' : (p.residualPx < 2.5 ? '#dd2' : '#e22');
  return p.rejected ? '#e2e' : '#888';
}

function draw() {
  const r = viewer.container.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(r.width * dpr) || canvas.height !== Math.round(r.height * dpr)) {
    canvas.width = Math.round(r.width * dpr); canvas.height = Math.round(r.height * dpr);
    canvas.style.width = r.width + 'px'; canvas.style.height = r.height + 'px';
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, r.width, r.height);
  if (!main()) return;
  const zoom = main().viewportToImageZoom(viewer.viewport.getZoom(true)); // screen px per image px
  if (cb('cbFrames').checked) {
    ctx.lineWidth = 1; ctx.strokeStyle = 'rgba(0,200,255,.8)'; ctx.fillStyle = '#0cf';
    ctx.font = Math.max(9, Math.min(22, 200 * zoom)) + 'px monospace';
    for (const f of DATA.frames) {
      const [x0, y0] = toScreen(f.px, f.py);
      const [x1, y1] = toScreen(f.px + DATA.frameW, f.py + DATA.frameH);
      ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      const [cx, cy] = toScreen(...centre(f));
      ctx.fillText(f.name, cx + 4, cy - 4);
      ctx.fillRect(cx - 2, cy - 2, 4, 4);
    }
  }
  if (cb('cbPairs').checked) {
    ctx.font = '12px monospace';
    for (const p of DATA.pairs) {
      const [ax, ay] = toScreen(...centre(DATA.frames[p.a]));
      const [bx, by] = toScreen(...centre(DATA.frames[p.b]));
      ctx.strokeStyle = pairColor(p); ctx.lineWidth = p.valid ? 2.5 : 1;
      ctx.setLineDash(p.valid ? [] : [5, 4]);
      ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
      if (p.residualPx !== null && zoom > 0.04) {
        ctx.fillStyle = pairColor(p);
        ctx.fillText(p.valid ? p.residualPx.toFixed(1) : 'x', (ax + bx) / 2 + 3, (ay + by) / 2 - 3);
      }
    }
    ctx.setLineDash([]);
  }
  if (cb('cbRegions').checked) {
    ctx.lineWidth = 2; ctx.strokeStyle = '#f0f'; ctx.fillStyle = 'rgba(255,0,255,.12)'; ctx.font = '12px monospace';
    for (const r of DATA.regions) {
      ctx.beginPath();
      r.polygon.forEach(([x, y], n) => { const [sx, sy] = toScreen(x, y); n ? ctx.lineTo(sx, sy) : ctx.moveTo(sx, sy); });
      ctx.closePath(); ctx.fill(); ctx.stroke();
      const [sx, sy] = toScreen(r.polygon[0][0], r.polygon[0][1]);
      ctx.fillStyle = '#f0f'; ctx.fillText(DATA.frames[r.frame].name, sx + 3, sy - 3); ctx.fillStyle = 'rgba(255,0,255,.12)';
    }
  }
  if (DATA.stack && DATA.stack.marking && layerIdx === 'm' && cb('cbParts').checked) {
    ctx.lineWidth = 1; ctx.setLineDash([4, 3]); ctx.strokeStyle = 'rgba(255,220,0,.7)'; ctx.fillStyle = '#fd0';
    ctx.font = '11px monospace';
    for (const p of DATA.stack.marking.parts) {
      const [x0, y0] = toScreen(p.block[0], p.block[1]);
      const [x1, y1] = toScreen(p.block[0] + p.block[2], p.block[1] + p.block[3]);
      ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      ctx.fillText(p.frame ? p.frame + ' Z' + p.z.toFixed(1) : 'no whole-part view', x0 + 2, y0 - 3);
    }
    ctx.setLineDash([]);
  }
  drawTops();
  drawCad(zoom);
  if (picked) {
    const [px, py] = toScreen(picked.x, picked.y);
    ctx.strokeStyle = '#f0f'; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(px, py, 9, 0, 2 * Math.PI); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(px - 15, py); ctx.lineTo(px + 15, py); ctx.moveTo(px, py - 15); ctx.lineTo(px, py + 15); ctx.stroke();
  }
}

function setSeams(on) {
  if (on && !seamsItem) {
    viewer.addTiledImage({ tileSource: { type: 'image', url: 'seams.png' }, x: 0, y: 0, width: 1, opacity: 0.9,
                           success: e => { seamsItem = e.item; } });
  } else if (!on && seamsItem) {
    viewer.world.removeItem(seamsItem); seamsItem = null;
  }
}

function frameAt(x, y) {
  const hits = [];
  for (const f of DATA.frames) {
    if (x >= f.px && x < f.px + DATA.frameW && y >= f.py && y < f.py + DATA.frameH) {
      const [cx, cy] = centre(f);
      hits.push({ f, d: Math.hypot(x - cx, y - cy) });
    }
  }
  hits.sort((a, b) => a.d - b.d);
  // inside a parallax region the panorama uses that region's frame, not the nearest centre
  const r = DATA.regions.find(r => inPolygon(x, y, r.polygon));
  if (r) {
    const n = hits.findIndex(h => h.f === DATA.frames[r.frame]);
    if (n > 0) hits.unshift(hits.splice(n, 1)[0]);
  }
  return hits;
}

function inPolygon(x, y, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

let picked = null;           // panorama point of the last click (work px)
let cropZoom = 1;            // source px -> screen px in the inspector crops
let showWhole = false;
const srcW = DATA.frameW / s, srcH = DATA.frameH / s;   // original frame size

function renderInfo() {
  const info = cb('info');
  if (!picked) { info.style.display = 'none'; return; }
  const hits = frameAt(picked.x, picked.y);
  if (!hits.length) { info.style.display = 'none'; return; }
  const zooms = [0.5, 1, 2, 3];
  let h = '<div class="head"><div><b>' + hits.length + ' source frame' + (hits.length > 1 ? 's' : '') +
    '</b> contain this point <small>(panorama ' + (picked.x / s).toFixed(0) + ', ' + (picked.y / s).toFixed(0) +
    ')' + (heightAt(picked.x, picked.y) !== null ? ', height ' + heightAt(picked.x, picked.y).toFixed(2) + ' mm' : '') +
    '</small></div><div>' + zooms.map(z => '<button data-z="' + z + '" class="' + (z === cropZoom ? 'on' : '') +
    '">' + z + 'x</button>').join(' ') + ' <button id="btnWhole" class="' + (showWhole ? 'on' : '') +
    '">frame</button> <button id="btnClose">&times;</button></div></div>' +
    '<small>green = used in the panorama, the rest sorted by distance from the frame centre. Originals, no corrections. ' +
    'Click a crop to open the full frame.</small><div id="cards">';
  hits.forEach((hit, n) => {
    const f = hit.f;
    const u = (picked.x - f.px) / s, v = (picked.y - f.py) / s;   // original frame px
    const bw = srcW * cropZoom, bh = srcH * cropZoom;
    h += '<div class="card' + (n === 0 ? ' used' : '') + '"><b>' + f.name + '</b>' + (n === 0 ? '<span class="tag">used</span>' : '') +
      ' <small>X ' + f.x + ' Y ' + f.y + ' mm &middot; px ' + u.toFixed(0) + ', ' + v.toFixed(0) +
      ' &middot; ' + (hit.d / s).toFixed(0) + ' px from centre</small>' +
      '<div class="crop" data-src="sources/' + f.name + '.jpg"><img src="sources/' + f.name + '.jpg" style="width:' +
      bw + 'px;height:' + bh + 'px;left:calc(50% - ' + (u * cropZoom) + 'px);top:calc(50% - ' + (v * cropZoom) +
      'px)"></div>' +
      (showWhole ? '<div class="whole"><img src="frames/' + f.name + '.jpg"><i style="left:' + (u / srcW * 100) +
        '%;top:' + (v / srcH * 100) + '%"></i></div>' : '') +
      '</div>';
  });
  info.innerHTML = h + '</div>';
  info.style.display = 'block';
  info.querySelectorAll('button[data-z]').forEach(b => b.onclick = () => { cropZoom = +b.dataset.z; renderInfo(); });
  cb('btnWhole').onclick = () => { showWhole = !showWhole; renderInfo(); };
  cb('btnClose').onclick = () => { picked = null; renderInfo(); draw(); };
  info.querySelectorAll('.crop').forEach(c => c.onclick = () => window.open(c.dataset.src, '_blank'));
}

viewer.addHandler('canvas-click', e => {
  if (!e.quick || !main()) return;
  const img = main().viewerElementToImageCoordinates(e.position);
  const p = partAt(img.x, img.y);   // a CAD part when the board layer is on, else the frames inspector
  if (p) { part = p; picked = null; renderPart(); draw(); return; }
  part = null;
  picked = { x: img.x, y: img.y };
  renderInfo();
  draw();
});

// ---------------------------------------------------------------- focus stack
const ST = DATA.stack;
let heights = null, heightItem = null, layerItems = {}, layerIdx = -1;
function heightAt(x, y) {
  if (!heights) return null;
  const c = Math.floor(x / ST.cell), r = Math.floor(y / ST.cell);
  if (c < 0 || r < 0 || c >= ST.cells[0] || r >= ST.cells[1]) return null;
  return heights[r * ST.cells[0] + c] / 1000;
}
function showLayer(i) {             // -1 = all-in-focus (main item), 'm' = marking, 'f' = fused floors, else a height
  layerIdx = i;
  if (i !== 'f') { floorSel = typeof i === 'number' && i >= 0 ? new Set([i]) : null; floorChips(); setTimeout(topsChanged); }
  for (const k in layerItems) layerItems[k].setOpacity(k === String(i) ? 1 : 0);
  if (i === 'f') {
    const sel = floorSel, key = floorKey();
    if (layerItems.f && layerItems.f.floorKey === key) { layerItems.f.setOpacity(1); }
    else fusedSource(sel).then(src => { if (floorKey() !== key) return;
      viewer.addTiledImage({ tileSource: src, index: 1, x: 0, y: 0, width: 1, success: e => {
        if (layerItems.f) viewer.world.removeItem(layerItems.f);
        layerItems.f = e.item; e.item.floorKey = key; e.item.setOpacity(layerIdx === 'f' ? 1 : 0); } }); });
    document.querySelectorAll('input[name=layer]').forEach(r => r.checked = false);
    draw();
    return;
  }
  if (i !== -1 && !layerItems[i]) {
    viewer.addTiledImage({ tileSource: i === 'm' ? ST.marking.tiles : ST.layers[i].tiles, index: 1, x: 0, y: 0, width: 1,
      success: e => { layerItems[i] = e.item; e.item.setOpacity(layerIdx === i ? 1 : 0); } });
  }
  document.querySelector('input[name=layer][value="' + (i === 'm' ? 'm' : i < 0 ? -1 : 'z') + '"]').checked = true;
  if (i !== 'm' && i >= 0) cb('zSlider').value = i;
  draw();
  const L = ST.layers[+cb('zSlider').value];
  cb('zLabel').textContent = 'Z ' + L.z.toFixed(1) + ' (' + (L.aboveBoardMm >= 0 ? '+' : '') +
    L.aboveBoardMm.toFixed(1) + ' mm)';
}
function turbo(x) {                // Google turbo colour map, polynomial fit, x in 0..1 -> [r, g, b] 0..255
  const r = 0.13572138 + x * (4.61539260 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943))));
  const g = 0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604))));
  const b = 0.10667330 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973))));
  return [r, g, b].map(v => Math.max(0, Math.min(255, Math.round(v * 255))));
}
let measured = null;
function heightImage(vmax) {        // data URL: turbo 0..vmax, filled (unmeasured) cells faint
  const [w, h] = ST.cells, c = document.createElement('canvas');
  c.width = w; c.height = h;
  const ctx2 = c.getContext('2d'), img = ctx2.createImageData(w, h), lut = [];
  for (let i = 0; i < 256; i++) lut.push(turbo(i / 255));
  for (let n = 0; n < w * h; n++) {
    const t = lut[Math.max(0, Math.min(255, Math.round(heights[n] / 1000 / vmax * 255)))];
    img.data[4 * n] = t[0]; img.data[4 * n + 1] = t[1]; img.data[4 * n + 2] = t[2];
    img.data[4 * n + 3] = measured && !measured[n] ? 70 : 255;
  }
  ctx2.putImageData(img, 0, 0);
  return c.toDataURL('image/png');
}
function setHeightOverlay(on) {
  if (heightItem) { viewer.world.removeItem(heightItem); heightItem = null; }
  if (on && heights) {
    const vmax = +cb('hRange').value;
    viewer.addTiledImage({ tileSource: { type: 'image', url: heightImage(vmax) }, x: 0, y: 0,
      width: ST.cells[0] * ST.cell / DATA.width, opacity: cb('hOpacity').value / 100,
      success: e => { if (heightItem) viewer.world.removeItem(heightItem); heightItem = e.item; } });
  }
  cb('hLegend').style.display = on ? 'block' : 'none';
  legend();
}
function legend() {
  const vmax = +cb('hRange').value;
  cb('hRangeV').textContent = vmax.toFixed(1);
  cb('hLegend').innerHTML = '<div style="height:8px;border-radius:2px;background:linear-gradient(90deg,' +
    [0, 0.17, 0.33, 0.5, 0.67, 0.83, 1].map(t => 'rgb(' + turbo(t).join(',') + ')').join(',') +
    ')"></div><small style="display:flex;justify-content:space-between"><span>0</span><span>' +
    (vmax / 2).toFixed(2) + '</span><span>&ge; ' + vmax.toFixed(1) + ' mm</span></small>' +
    '<small>faint = not measured (no texture), filled from around</small>';
}
// ---------------------------------------------------------------- floors: the board from any set of heights
// One floor = that height's panorama; all = the all-in-focus panorama; several = focus fusion of those heights only,
// made here per tile: weights exp(-(Zfocus - z)^2 / 2 sigma^2) over the picked heights (as FocusStack.fuse), Zfocus
// per cell from focusz.bin (focus_z.npy, blurred like fuse).
let floorSel = null, focusZ = null, fusedN = 0;
function floorKey() { return floorSel ? [...floorSel].sort((a, b) => a - b).join(',') : 'all'; }
function floorChips() {
  cb('floorChips').innerHTML = ST.layers.map((l, i) => '<button class="zc' + (!floorSel || floorSel.has(i) ? ' on' : '') +
    '" data-i="' + i + '">' + l.z.toFixed(1) + '</button>').join('') +
    ' <button class="zc' + (!floorSel ? ' on' : '') + '" data-i="all">all</button>';
  cb('floorChips').querySelectorAll('button').forEach(e => e.onclick = ev => {
    const i = e.dataset.i;
    if (i === 'all') floorSel = null;
    else if (ev.ctrlKey || ev.metaKey) {
      floorSel = floorSel ? new Set(floorSel) : new Set(ST.layers.map((_, k) => k));
      floorSel.has(+i) ? floorSel.delete(+i) : floorSel.add(+i);
      if (!floorSel.size || floorSel.size === ST.layers.length) floorSel = null;
    } else floorSel = new Set([+i]);
    showFloors();
  });
}
function showFloors() {
  floorChips();
  topsChanged();
  if (!floorSel) return showLayer(-1);
  if (floorSel.size === 1) return showLayer([...floorSel][0]);
  showLayer('f');
}
function dziInfo(url) {
  return fetch(url).then(r => r.text()).then(t => {
    const x = new DOMParser().parseFromString(t, 'text/xml'), im = x.documentElement, sz = x.getElementsByTagName('Size')[0];
    return { w: +sz.getAttribute('Width'), h: +sz.getAttribute('Height'), ts: +im.getAttribute('TileSize'),
             ov: +im.getAttribute('Overlap'), fmt: im.getAttribute('Format') };
  });
}
function loadImg(src) {
  return new Promise((ok, bad) => { const i = new Image(); i.onload = () => ok(i); i.onerror = () => bad(src); i.src = src; });
}
function fusedSource(sel) {
  const idx = [...sel].sort((a, b) => a - b), zs = idx.map(i => ST.layers[i].z), sig = ST.depthSigma || 0.3;
  const dirs = idx.map(i => ST.layers[i].tiles.replace(/\.dzi$/, '_files/'));
  return Promise.all([dziInfo(ST.layers[idx[0]].tiles), focusZ ? null :
    fetch('focusz.bin').then(r => r.arrayBuffer()).then(b => { focusZ = new Int16Array(b); })]).then(([d]) => {
    const src = new OpenSeadragon.DziTileSource(d.w, d.h, d.ts, d.ov, dirs[0], d.fmt);
    const maxLevel = src.maxLevel, key = 'fused' + (++fusedN) + '[' + zs.join(',') + ']';
    src.getTileUrl = (level, x, y) => key + '/' + level + '/' + x + '_' + y;
    src.downloadTileAbort = () => {};
    src.downloadTileStart = ctx => {
      const [, lv, xy] = ctx.src.split('/'), [x, y] = xy.split('_').map(Number), level = +lv;
      Promise.all(dirs.map(dd => loadImg(dd + level + '/' + x + '_' + y + '.' + d.fmt))).then(ims => {
        const w = ims[0].naturalWidth, h = ims[0].naturalHeight, c = document.createElement('canvas');
        c.width = w; c.height = h;
        const g = c.getContext('2d', { willReadFrequently: true }), px = ims.map(im => {
          g.drawImage(im, 0, 0); return g.getImageData(0, 0, w, h).data; });
        const out = g.createImageData(w, h), o = out.data, k = 2 ** (maxLevel - level);   // image px per level px
        const x0 = x * d.ts - (x ? d.ov : 0), y0 = y * d.ts - (y ? d.ov : 0), [cw, chh] = ST.cells, cs = ST.cell;
        const wk = new Float32Array(zs.length);
        for (let v = 0; v < h; v++) {
          const cy = Math.min(chh - 1, Math.max(0, ((y0 + v + 0.5) * k) / cs - 0.5)), r0 = Math.floor(cy), fy = cy - r0, r1 = Math.min(chh - 1, r0 + 1);
          for (let u = 0; u < w; u++) {
            const cx = Math.min(cw - 1, Math.max(0, ((x0 + u + 0.5) * k) / cs - 0.5)), c0 = Math.floor(cx), fx = cx - c0, c1 = Math.min(cw - 1, c0 + 1);
            const t = ((focusZ[r0 * cw + c0] * (1 - fx) + focusZ[r0 * cw + c1] * fx) * (1 - fy) +
                       (focusZ[r1 * cw + c0] * (1 - fx) + focusZ[r1 * cw + c1] * fx) * fy) / 1000;
            let dmin = Infinity;
            for (let a = 0; a < zs.length; a++) { wk[a] = (t - zs[a]) ** 2; if (wk[a] < dmin) dmin = wk[a]; }
            let tot = 0;
            for (let a = 0; a < zs.length; a++) { wk[a] = Math.exp(-0.5 * (wk[a] - dmin) / (sig * sig)); tot += wk[a]; }
            const n = 4 * (v * w + u);
            for (let ch = 0; ch < 3; ch++) {
              let acc = 0;
              for (let a = 0; a < zs.length; a++) acc += px[a][n + ch] * wk[a];
              o[n + ch] = acc / tot;
            }
            o[n + 3] = 255;
          }
        }
        g.putImageData(out, 0, 0);
        ctx.finish(c, null);
      }).catch(e => ctx.finish(null, null, 'fused tile: ' + e));
    };
    return src;
  });
}

// part tops from the source views (cutMarkingViews.py): a mosaic per part. Every frame that sees the part gives its
// sharpest view among the picked floors, clipped to the part outline AND to that frame's coverage (poly): the frames
// with less of the body go first, the frame the VLM read (vlmMarkings.py --views; else the most complete one) last,
// on top. Where no frame reaches, the board below shows. VIEWS[ref].W maps view px to image px.
const topImgs = {};
let tops = [];
function topImg(file) {
  if (!topImgs[file]) { topImgs[file] = new Image(); topImgs[file].onload = requestDraw; topImgs[file].src = BASE + file; }
  return topImgs[file];
}
function topsChanged() {
  tops = [];
  if (!cb('cbTops') || !Object.keys(VIEWS).length) return;
  cb('lblTops').style.display = 'block';
  if (!cb('cbTops').checked || !CAD) return requestDraw();
  const zs = ST && floorSel ? new Set([...floorSel].map(i => ST.layers[i].z)) : null;
  for (const [ref, vv] of Object.entries(VIEWS)) {
    const p = CAD.byRef[ref], m = markOf(ref);
    if (!p || !vv.W) continue;
    const best = {};
    for (const x of vv.all || vv.views) if ((!zs || zs.has(x.z)) && (!best[x.frame] || x.focus > best[x.frame].focus)) best[x.frame] = x;
    const views = Object.values(best);
    if (!views.length) continue;
    const want = m && m.vlmViews && (m.vlmViews.show || m.vlmViews.frame);
    const top = best[want] || views.reduce((a, b) => b.cover > a.cover ? b : a);
    const layers = views.filter(x => x !== top).sort((a, b) => a.cover - b.cover).concat([top]).map(x => {
      const clip = new Path2D();
      (x.poly || []).forEach(([u, v], n) => n ? clip.lineTo(u, v) : clip.moveTo(u, v));
      return { img: topImg(x.file), clip: x.poly && x.poly.length > 2 ? clip : null };
    });
    tops.push({ p, W: vv.W, layers });
  }
  requestDraw();
}
function drawTops() {
  if (!tops.length || !cb('cbTops').checked) return;
  const dpr = window.devicePixelRatio || 1, [a, b, c, d, e, f] = imgToScreenAffine();
  for (const t of tops) {
    const W = t.W;
    ctx.save();
    ctx.setTransform(dpr * a, dpr * b, dpr * c, dpr * d, dpr * e, dpr * f);
    ctx.clip(t.p.path);
    ctx.transform(W[0][0], W[1][0], W[0][1], W[1][1], W[0][2], W[1][2]);
    for (const l of t.layers) {
      if (!l.img.complete || !l.img.naturalWidth) continue;
      ctx.save();
      if (l.clip) ctx.clip(l.clip);
      ctx.drawImage(l.img, 0, 0);
      ctx.restore();
    }
    ctx.restore();
  }
}

if (ST) {
  cb('stackPanel').style.display = 'block';
  fetch('height.bin').then(r => r.arrayBuffer()).then(b => { heights = new Int16Array(b); });
  fetch('measured.bin').then(r => r.ok ? r.arrayBuffer() : null).then(b => { if (b) measured = new Uint8Array(b); });
  cb('hRange').addEventListener('input', legend);
  cb('hRange').addEventListener('change', () => setHeightOverlay(cb('cbHeight').checked));
  const tiled = ST.layers.filter(l => l.tiles).length === ST.layers.length;
  cb('zSlider').max = ST.layers.length - 1;
  cb('zSlider').value = ST.layers.findIndex(l => Math.abs(l.aboveBoardMm) < 0.3);
  if (!tiled) document.querySelector('input[name=layer][value="z"]').parentElement.parentElement.style.display = 'none';
  showLayer(-1);
  cb('zSlider').addEventListener('input', () => showLayer(+cb('zSlider').value));
  document.querySelectorAll('input[name=layer]').forEach(r => r.addEventListener('change', () => {
    const i = r.value === '-1' ? -1 : r.value === 'm' ? 'm' : +cb('zSlider').value;
    floorSel = i === -1 || i === 'm' ? null : new Set([i]); floorChips(); topsChanged(); showLayer(i); }));
  cb('zSlider').addEventListener('input', () => { floorSel = new Set([+cb('zSlider').value]); floorChips(); topsChanged(); });
  floorChips();
  cb('cbTops').addEventListener('change', topsChanged);
  if (ST.marking) cb('markingRow').style.display = 'flex';
  cb('cbParts').addEventListener('change', draw);
  cb('cbHeight').addEventListener('change', () => setHeightOverlay(cb('cbHeight').checked));
  cb('hOpacity').addEventListener('input', () => { if (heightItem) heightItem.setOpacity(cb('hOpacity').value / 100); });
  legend();
  window.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT' && e.target.type !== 'radio' && e.target.type !== 'checkbox') return;
    const n = ST.layers.length, onZ = typeof layerIdx === 'number' && layerIdx >= 0;
    if (e.key === ']' && tiled) showLayer(Math.min(n - 1, (onZ ? layerIdx + 1 : +cb('zSlider').value)));
    else if (e.key === '[' && tiled) showLayer(Math.max(0, (onZ ? layerIdx - 1 : +cb('zSlider').value)));
    else if (e.key === 'f') showLayer(-1);
    else if (e.key === 'm' && ST.marking) showLayer('m');
  });
  new OpenSeadragon.MouseTracker({ element: viewer.canvas, moveHandler: e => {
    if (!main()) return;
    const p = main().viewerElementToImageCoordinates(e.position);
    const h = heightAt(p.x, p.y);
    cb('hCursor').textContent = h === null ? '' : 'cursor: ' + h.toFixed(2) + ' mm above the board';
  } });
}

// ---------------------------------------------------------------- board (CAD) from alignBoard.py: board.json
const FIT = { fitted: '#4fbf73', nc: '#e5534b', optional: '#e0a33a', none: '#8b939c' };
const FIT_LABEL = { fitted: 'fitted', nc: 'NC — must be empty', optional: 'optional (X/NC)', none: 'not in BOM' };
const GROUP_COL = ['#4e9cf5', '#e0a33a', '#b07cf0', '#e5534b', '#3fc1b0', '#f07ac0', '#4fbf73', '#d9d65a', '#9aa5b1', '#f29a5c', '#6c7a89'];
const GROUP_ORDER = ['Resistors', 'Capacitors', 'Inductors', 'Diodes & protection', 'Transistors', 'LEDs & optocouplers', 'ICs',
                     'Crystals & oscillators', 'Connectors', 'Buttons & switches', 'Other', 'not in BOM'];
let CAD = null, cadPxPerMm = 77, part = null, cadHidden = new Set(), groupCol = {}, cadPaths = null;
// markings.json (markingSheets.py + markingCheck.py): text read on each part and its check against the BOM
let MARKS = {};
const BASE = DATA.boardBase || '';   // a focus-stack page shows the board of its reference panorama folder
let VIEWS = {};   // marking_views.json (cutMarkingViews.py): the part body cut out of every source frame that sees it
const CHECK_COL = { ok: '#4fbf73', mismatch: '#e5534b', missing: '#e5534b', populated: '#e0a33a', unread: '#9aa5b1', info: '#6cbcff' };
const PROBLEM = new Set(['mismatch', 'missing', 'populated']);
const OCR_COL = { same: '#4fbf73', close: '#d9d65a', partial: '#a6d96a', differs: '#e0a33a', 'ocr-only': '#e0a33a', none: '#666' };
function vlmBadge(m) {   // Qwen3-VL-2B second reading (vlmMarkings.py, its own venv)
  if (!m || !m.vlm) return '';
  const v = m.vlm, col = OCR_COL[v.agree] || '#999';
  const txt = v.agree === 'same' ? 'VLM ✓' : v.agree === 'none' ? 'VLM —' : v.agree === 'close' ? 'VLM ≈ ' + v.lines.join(' / ') :
    v.agree === 'partial' ? 'VLM ✓ part: ' + v.lines.join(' / ') : 'VLM: ' + (v.lines.join(' / ') || '—');
  return '<span style="color:' + col + '">' + esc(txt) + '</span>';
}
function viewsBadge(m) {   // VLM over every source view, voted (vlmMarkings.py --views)
  if (!m || !m.vlmViews) return '';
  const v = m.vlmViews, col = OCR_COL[v.agree] || '#999', t = v.lines.join(' / ');
  const txt = !v.lines.length ? 'views —' : v.agree === 'same' ? 'views ✓' : v.agree === 'ocr-only' ? 'views new: ' + t : 'views: ' + t;
  return '<span style="color:' + (v.agree === 'ocr-only' ? '#6cf' : col) + '">' + esc(txt) + ' <small>' + v.support + '/' + v.views + '</small></span>';
}
function codeBadge(m) {   // our resistor-code classifier (readResistorCodes.py), leave-one-out on parts in the dataset
  if (!m || !m.resistorCode) return '';
  const r = m.resistorCode;
  if (!r.code) return '<span style="color:#666">code —</span>';
  const col = r.agree === 'same' ? '#4fbf73' : r.bom === 'mismatch' ? '#e5534b' : '#e0a33a';
  return '<span style="color:' + col + '">' + (r.agree === 'same' ? 'code ✓' : 'code: ' + esc(r.code)) + '</span>';
}
function ocrBadge(m) {   // second, independent reading by the local OCR (ocrMarkings.py)
  if (!m || !m.ocr) return '';
  const o = m.ocr, alt = o.otherTurn;
  const txt = o.agree === 'same' ? 'OCR ✓' : o.agree === 'none' ? 'OCR —' : o.agree === 'partial' ? 'OCR ✓ part: ' + o.lines.join(' / ') : alt ? 'OCR ✓ (turned ' + alt.turn + '°)' :
    'OCR: ' + (o.lines.join(' / ') || '—');
  return '<span style="color:' + (alt ? OCR_COL[alt.agree] : OCR_COL[o.agree]) + '">' + esc(txt) + '</span>';
}
function markOf(ref) { return MARKS[ref] || null; }
const esc = t => String(t ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
// align.M maps board mm to px of the image it was fitted on (panorama.jpg = the viewer's image)
function mmToImg(M, x, y) { return [M[0][0] * x + M[0][1] * y + M[0][2], M[1][0] * x + M[1][1] * y + M[1][2]]; }
function pathPoints(d) { return (d.match(/[ML][^MLZ]+/g) || []).map(t => t.slice(1).trim().split(/[ ,]+/).map(Number)); }
function lineOf(c) { return c.line == null ? null : CAD.board.lines[c.line]; }
function groupOf(line) { return line ? line.cat.group : 'not in BOM'; }
function cadKey(p) { return cb('cadMode').value === 'fit' ? p.c.fit : groupOf(p.line); }
function cadColour(p) { return cb('cadMode').value === 'fit' ? FIT[p.c.fit] : (p.line ? groupCol[p.line.cat.group] : FIT.none); }
function toPath(polys) {
  const P = new Path2D();
  for (const q of polys) { q.forEach(([x, y], n) => n ? P.lineTo(x, y) : P.moveTo(x, y)); P.closePath(); }
  return P;
}
const natKey = t => t.split(/(\d+)/).map(x => /^\d+$/.test(x) ? x.padStart(6, '0') : x.toLowerCase()).join('');

fetch(BASE + 'board.json', { cache: 'no-cache' }).then(r => r.ok ? r.json() : null).then(b => {   // revalidate: alignBoard.py rewrites it
  if (!b || !b.align) return;
  const M = b.align.M, side = b.align.side;
  CAD = { board: b, parts: [], byRef: {}, crops: b.align.crops || {} };
  cadPxPerMm = (b.align.pxPerMm[0] + b.align.pxPerMm[1]) / 2;
  [...new Set(b.lines.map(l => l.cat.group))].forEach((g, i) => groupCol[g] = GROUP_COL[i % GROUP_COL.length]);
  for (const [ref, c] of Object.entries(b.components)) {
    if (c.side !== side) continue;
    const pts = c.outline ? pathPoints(c.outline) : [[c.bb[0], c.bb[1]], [c.bb[2], c.bb[1]], [c.bb[2], c.bb[3]], [c.bb[0], c.bb[3]]];
    const poly = pts.map(([x, y]) => mmToImg(M, x, y));
    const pads = c.pins.filter(p => p.side === side || p.side === 'both').map(p =>
      [[p.bb[0], p.bb[1]], [p.bb[2], p.bb[1]], [p.bb[2], p.bb[3]], [p.bb[0], p.bb[3]]].map(([x, y]) => mmToImg(M, x, y)));
    const xs = poly.map(q => q[0]), ys = poly.map(q => q[1]);
    const box = [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
    const p = { ref, c, line: lineOf(c), poly, path: toPath([poly]), padPath: toPath(pads), box,
                cx: (box[0] + box[2]) / 2, cy: (box[1] + box[3]) / 2, area: (box[2] - box[0]) * (box[3] - box[1]) };
    CAD.parts.push(p); CAD.byRef[ref] = p;
  }
  const f = b.align.fine, um = 1000 / cadPxPerMm;
  cb('cadInfo').textContent = (b.source.board || b.source.project) + ' r' + b.source.rev + ' · ' + side + ' · fit ' +
    (f.medianPx * um).toFixed(0) + ' µm median (' + f.inliers + ' holes)';
  cb('btnList').textContent = 'parts list (' + Object.keys(b.components).length + ')';
  cb('cadPanel').style.display = 'block';
  cadChanged();
  return fetch(BASE + 'markings.json', { cache: 'no-cache' }).then(r => r.ok ? r.json() : null).then(m => {
    if (!m) return;
    MARKS = m.parts || {};
    fetch(BASE + 'marking_views.json', { cache: 'no-cache' }).then(r => r.ok ? r.json() : null).then(v => {
      VIEWS = v || {}; if (part) renderPart(); topsChanged(); }).catch(() => {});
    const bad = Object.values(MARKS).filter(x => x.check && PROBLEM.has(x.check.status)).length;
    cb('findCount').textContent = '(' + bad + ')';
    cb('lblFind').hidden = false;
    cadChanged();
    if (cb('list').style.display === 'block') renderList();
  });
}).catch(e => console.error('board.json / markings.json', e));

// colour / visibility changed: rebuild the cached paths (one Path2D per colour, one for all pads)
function cadChanged() {
  const by = new Map(), pads = new Path2D();
  for (const p of CAD.parts) {
    if (cadHidden.has(cadKey(p))) continue;
    const id = cadColour(p) + (p.c.fit === 'nc' ? '|dash' : '|solid');
    if (!by.has(id)) by.set(id, { col: cadColour(p), dash: p.c.fit === 'nc', path: new Path2D() });
    by.get(id).path.addPath(p.path);
    pads.addPath(p.padPath);
  }
  const problems = new Path2D();
  for (const p of CAD.parts) { const m = markOf(p.ref); if (m && m.check && PROBLEM.has(m.check.status)) problems.addPath(p.path); }
  cadPaths = { groups: [...by.values()], pads, problems };
  cadLegend();
  requestDraw();
}

function cadLegend() {
  const counts = {};
  for (const p of CAD.parts) counts[cadKey(p)] = (counts[cadKey(p)] || 0) + 1;
  const items = cb('cadMode').value === 'fit' ? Object.keys(FIT).map(k => [k, FIT[k], FIT_LABEL[k]])
    : [...Object.keys(groupCol).map(g => [g, groupCol[g], g]), ['not in BOM', FIT.none, 'not in BOM']];
  cb('cadLegend').innerHTML = items.filter(([k]) => counts[k]).map(([k, c, l]) =>
    '<span class="k' + (cadHidden.has(k) ? ' off' : '') + '" data-k="' + esc(k) + '"><i style="background:' + c + '"></i>' +
    esc(l) + ' <small>' + counts[k] + '</small></span>').join('');
  cb('cadLegend').querySelectorAll('span.k').forEach(e => e.onclick = () => {
    const k = e.dataset.k; cadHidden.has(k) ? cadHidden.delete(k) : cadHidden.add(k); cadChanged(); });
}

// image px -> screen px of the main image (OSD has no rotation here): [a b c d e f] as in setTransform
function imgToScreenAffine() {
  const [x0, y0] = toScreen(0, 0), [x1, y1] = toScreen(1000, 0), [x2, y2] = toScreen(0, 1000);
  return [(x1 - x0) / 1000, (y1 - y0) / 1000, (x2 - x0) / 1000, (y2 - y0) / 1000, x0, y0];
}

function drawCad(zoom) {
  if (!CAD || !cadPaths) return;
  const showParts = cb('cbCad').checked, showPads = cb('cbPads').checked;   // independent layers
  if (!showParts && !showPads && !part && !cb('cbFindings').checked) return;
  const dpr = window.devicePixelRatio || 1, [a, b, c, d, e, f] = imgToScreenAffine(), k = Math.hypot(a, b);
  ctx.save();
  ctx.setTransform(dpr * a, dpr * b, dpr * c, dpr * d, dpr * e, dpr * f);   // stroke cached paths in image px
  if (showPads) { ctx.setLineDash([]); ctx.lineWidth = 1 / k; ctx.strokeStyle = 'rgba(0,230,255,.75)'; ctx.stroke(cadPaths.pads); }
  if (showParts) for (const g of cadPaths.groups) {
    ctx.lineWidth = 1.5 / k; ctx.setLineDash(g.dash ? [6 / k, 4 / k] : []); ctx.strokeStyle = g.col; ctx.stroke(g.path);
  }
  if (cb('cbFindings').checked && !cb('lblFind').hidden) {   // parts that disagree with the BOM
    ctx.setLineDash([]); ctx.lineWidth = Math.max(4 / k, 0.6 * cadPxPerMm * 0.15); ctx.strokeStyle = 'rgba(255,60,60,.85)';
    ctx.stroke(cadPaths.problems); ctx.fillStyle = 'rgba(255,60,60,.18)'; ctx.fill(cadPaths.problems);
  }
  if (part) {
    ctx.setLineDash([]); ctx.lineWidth = 3 / k; ctx.strokeStyle = '#fff'; ctx.stroke(part.path);
    ctx.fillStyle = 'rgba(255,255,255,.12)'; ctx.fill(part.path);
  }
  ctx.restore();
  // names: screen space, only parts in view, only once they are big enough
  if (showParts && zoom * cadPxPerMm > 14) {
    const v = main().viewportToImageRectangle(viewer.viewport.getBounds(true));
    ctx.font = '12px monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    for (const p of CAD.parts) {
      if (p.box[2] < v.x || p.box[0] > v.x + v.width || p.box[3] < v.y || p.box[1] > v.y + v.height) continue;
      if (cadHidden.has(cadKey(p))) continue;
      const sx = a * p.cx + c * p.cy + e, sy = b * p.cx + d * p.cy + f, w = p.ref.length * 7.3 + 6;
      ctx.fillStyle = 'rgba(0,0,0,.55)'; ctx.fillRect(sx - w / 2, sy - 8, w, 16);
      ctx.fillStyle = cadColour(p); ctx.fillText(p.ref, sx, sy);
    }
    ctx.textAlign = 'start'; ctx.textBaseline = 'alphabetic';
  }
  if (HUD.on) HUD.cadStats = { pads: showPads, parts: showParts, groups: cadPaths.groups.length };
}

function partAt(x, y) {
  if (!CAD || !cb('cbCad').checked) return null;
  const hits = CAD.parts.filter(p => !cadHidden.has(cadKey(p)) && x >= p.box[0] && x <= p.box[2] && y >= p.box[1] && y <= p.box[3]
    && inPolygon(x, y, p.poly));
  hits.sort((a, b) => a.area - b.area);
  return hits[0] || null;
}

// minimal markdown for datasheet sections: headings, tables, images, paragraphs, bold/code
function md(text, dir) {
  const inline = t => esc(t).replace(/\*\*(.+?)\*\*/g, '<b>$1</b>').replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/!\[([^\]]*)\]\(([^)\s]+)\)/g, (_, a, src) => '<img alt="' + a + '" src="' + (/^https?:/.test(src) ? src : dir + '/' + src.replace(/&amp;/g, '&')) + '">');
  const out = [], lines = text.split('\n');
  for (let i = 0; i < lines.length; i++) {
    const ln = lines[i]; let m;
    if ((m = ln.match(/^(#{2,6})\s+(.*)$/))) { const h = Math.min(6, m[1].length + 1); out.push('<h' + h + '>' + inline(m[2]) + '</h' + h + '>'); continue; }
    if (ln.trim().startsWith('|')) {
      const rows = []; while (i < lines.length && lines[i].trim().startsWith('|')) rows.push(lines[i++]); i--;
      const cells = r => r.trim().replace(/^\||\|$/g, '').split('|').map(c => c.trim());
      out.push('<table>' + rows.filter(r => !/^\s*\|[\s:|-]+\|\s*$/.test(r)).map((r, j) => '<tr>' +
        cells(r).map(c => j ? '<td>' + inline(c) + '</td>' : '<th>' + inline(c) + '</th>').join('') + '</tr>').join('') + '</table>');
      continue;
    }
    if (ln.trim()) out.push('<p>' + inline(ln) + '</p>');
  }
  return out.join('');
}

function renderPart() {
  const info = cb('info');
  if (!part) { info.style.display = 'none'; return; }
  const c = part.c, l = part.line, ds = l && l.ds;
  const lab = { zero: '0 Ω', 3: '3-digit', 4: '4-digit', eia96: 'EIA-96' };
  const codes = l && l.value && l.value.marking ? Object.entries(l.value.marking).map(([k, v]) => '<code title="' + lab[k] + '">' + esc(v) + '</code>').join('') +
    '<small>' + Object.keys(l.value.marking).map(k => lab[k]).join(' / ') + '</small>' : '';
  const dir = ds ? BASE + ds.file.replace(/\/[^/]*$/, '') : '';
  const secs = ds ? Object.entries(ds.sections).map(([kind, list]) => list.map((x, i) =>
    '<details' + (kind === 'marking' && i === 0 ? ' open' : '') + '><summary>' + esc(kind) + ': ' + esc(x.title) +
    (x.page ? ' <small>p. ' + esc(x.page) + '</small>' : '') + '</summary><div class="md">' + md(x.text, dir) + '</div></details>').join('')).join('') : '';
  info.innerHTML = '<div class="part"><div class="head"><div><b style="font-size:15px">' + esc(part.ref) + '</b> ' +
    '<span class="fit" style="color:' + FIT[c.fit] + '">' + esc(FIT_LABEL[c.fit]) + '</span></div><div>' +
    '<button id="btnFrames">frames</button> <button id="btnClose">&times;</button></div></div><div class="kv">' +
    (l ? '<b>BOM name</b><span>' + esc(l.name) + (l.fittedName && l.fittedName !== l.name ? ' <small>→ ' + esc(l.fittedName) + '</small>' : '') + '</span>' +
      '<b>type</b><span>' + esc(l.cat.group) + ' · ' + esc(l.cat.sub) + '</span>' +
      (l.value ? '<b>value</b><span>' + esc(l.value.text) + (l.value.voltage ? ' · ' + l.value.voltage + ' V' : '') +
        (l.value.dielectric ? ' · ' + esc(l.value.dielectric) : '') + (l.value.current ? ' · ' + l.value.current + ' A' : '') + '</span>' : '') +
      (codes ? '<b>marking</b><span>' + codes + '</span>' : '') +
      '<b>description</b><span>' + esc(l.description || '—') + '</span>' +
      (l.designators.length > 1 ? '<b>same line</b><span><small>' + esc(l.designators.join(', ')) + '</small></span>' : '') : '') +
    '<b>package</b><span>' + esc(c.package) + '</span><b>position</b><span>' + c.x + ', ' + c.y + ' mm · ' + c.rot + '°</span>' +
    '<b>height</b><span>' + (c.height ? c.height + ' mm' : '—') + '</span>' +
    (markOf(part.ref) ? (() => { const m = markOf(part.ref), st = m.check ? m.check.status : '';
      return '<b>marking read</b><span>' + (m.lines.length ? '<span class="mkread">' + m.lines.map(esc).join('<br>') + '</span>' :
        '<small>' + (m.present === false ? 'place empty' : 'nothing read') + '</small>') +
        ' <small>' + esc(m.conf) + (m.note ? ' · ' + esc(m.note) : '') + '</small>' +
        (st ? '<br><span style="color:' + CHECK_COL[st] + '">' + esc(st) + (m.check.text ? ': ' + esc(m.check.text) : '') + '</span>' : '') +
        '</span>'; })() : '') +
    (markOf(part.ref) && markOf(part.ref).ocr ? (() => { const o = markOf(part.ref).ocr;
      return '<b>OCR</b><span>' + (o.lines.length ? '<span class="mkread" style="color:#9fd">' + o.lines.map(esc).join('<br>') + '</span>' : '<small>nothing</small>') +
        ' <small>score ' + o.score.toFixed(2) + ', turn ' + o.turn + '°</small> ' + ocrBadge(markOf(part.ref)) +
        (o.otherTurn ? '<br><small>at ' + o.otherTurn.turn + '°: ' + esc(o.otherTurn.lines.join(' / ')) + '</small>' : '') + '</span>'; })() : '') +
    (markOf(part.ref) && markOf(part.ref).vlm ? (() => { const v = markOf(part.ref).vlm;
      return '<b>VLM</b><span>' + (v.lines.length ? '<span class="mkread" style="color:#c9f">' + v.lines.map(esc).join('<br>') + '</span>' : '<small>nothing</small>') +
        ' <small>score ' + v.score.toFixed(2) + ', turn ' + v.turn + '°, ' + esc(v.model.split('/').pop()) + '</small> ' + vlmBadge(markOf(part.ref)) +
        '</span>'; })() : '') +
    (markOf(part.ref) && markOf(part.ref).vlmViews ? (() => { const v = markOf(part.ref).vlmViews;
      return '<b>VLM views</b><span>' + (v.lines.length ? '<span class="mkread" style="color:#6cf">' + v.lines.map(esc).join('<br>') + '</span>' : '<small>nothing</small>') +
        ' <small>' + (v.lines.length ? 'score ' + v.score.toFixed(2) + ', ' + esc(v.frame) + ' turn ' + v.turn + '°, ' : '') +
        v.support + ' of ' + v.views + ' views agree, ' + esc(v.model.split('/').pop()) + '</small> ' + viewsBadge(markOf(part.ref)) + '</span>'; })() : '') +
    (markOf(part.ref) && markOf(part.ref).resistorCode ? (() => { const r = markOf(part.ref).resistorCode;
      return '<b>resistor code</b><span>' + (r.code ? '<span class="mkread" style="color:#fc9">' + esc(r.code) + '</span>' : '<small>not read</small>') +
        ' <small>score ' + (r.score ?? 0).toFixed(2) + (r.margin != null ? ', margin ' + r.margin.toFixed(2) : '') +
        (r.runnerUp ? ', runner-up ' + esc(r.runnerUp) : '') + ', turn ' + r.turn + '°' + (r.value ? ' = ' + esc(r.value) : '') + '</small> ' +
        codeBadge(markOf(part.ref)) + (r.bom === 'mismatch' ? ' <span style="color:#e5534b">BOM mismatch</span>' : '') +
        '<br><small>own classifier (readResistorCodes.py), trained without this part</small></span>'; })() : '') +
    '</div>' +
    (CAD.crops[part.ref] ? '<div class="imgs"><figure><img src="' + esc(BASE + CAD.crops[part.ref].file) + '"><figcaption>crop as cut: the part frame, ' +
      CAD.crops[part.ref].pxPerMm.toFixed(0) + ' px/mm</figcaption></figure>' +
      (markOf(part.ref) && markOf(part.ref).img ? '<figure><img src="' + esc(BASE + markOf(part.ref).img) + '"><figcaption>reading image: grey (L), CLAHE, ' +
        'enlarged, turned ' + markOf(part.ref).turn + '° so the text stands upright — what the marking was read from</figcaption></figure>' : '') +
      '</div>' : '') +
    (VIEWS[part.ref] && VIEWS[part.ref].views.length ? '<div id="viewsBlock">' + viewsHtml(part.ref) + '</div>' : '') +
    (ds ? '<h4 style="margin:12px 0 4px">Datasheet</h4><a href="' + esc(BASE + ds.file) + '" target="_blank">' + esc(ds.title) + '</a> <small>' +
      esc(ds.vendor) + ' · ' + esc(ds.status) + ' · by ' + esc(ds.method) + (ds.mpn ? ' ' + esc(ds.mpn) : '') + '</small>' +
      (ds.excursus ? '<p><b>' + esc(ds.excursus.category) + '</b> <small>' + esc(ds.excursus.params.join(' · ')) + '</small></p>' : '') +
      (secs || '<p><small>No marking / ordering / package sections.</small></p>')
      : (l ? '<p><small>No datasheet in the library.</small></p>' : '')) + '</div>';
  info.style.display = 'block';
  cb('btnClose').onclick = () => { part = null; renderPart(); draw(); };
  cb('btnFrames').onclick = () => { picked = { x: part.cx, y: part.cy }; part = null; renderInfo(); draw(); };
  bindViews();
}

// Source views of a part (cutMarkingViews.py): every frame that sees it, at the heights picked here. One height =
// what that layer's frames show (the single-height panorama is one of them); several = per frame the sharpest of
// them; all = per frame the sharpest overall (what vlmMarkings.py --views read). The pick stays across parts.
let viewZ = null;   // Set of picked heights, null = all
function viewsHtml(ref) {
  const vv = VIEWS[ref], m = markOf(ref), r = m && m.vlmViews;
  const all = vv.all || vv.views, zs = [...new Set(all.map(x => x.z))].sort((a, b) => a - b);
  const on = z => !viewZ || viewZ.has(z);
  const byFrame = {};
  for (const x of all) if (on(x.z) && (!byFrame[x.frame] || x.focus > byFrame[x.frame].focus)) byFrame[x.frame] = x;
  const order = vv.views.map(x => x.frame).filter(f => byFrame[f]);
  const readZ = Object.fromEntries(vv.views.map(x => [x.frame, x.z]));   // the height the VLM read per frame
  const chips = zs.map(z => '<button class="zc' + (on(z) ? ' on' : '') + '" data-z="' + z + '">' + z.toFixed(1) + '</button>').join('') +
    ' <button class="zc' + (!viewZ ? ' on' : '') + '" data-z="all">all</button>';
  return '<h4 style="margin:12px 0 4px">Source views <small>' + order.length + ' frames · ' +
    (!viewZ ? 'all heights, sharpest per frame' : viewZ.size === 1 ? 'height ' + [...viewZ][0].toFixed(1) + ' only' :
      viewZ.size + ' heights, sharpest per frame') + ' — the marking shows at some angles only</small></h4>' +
    '<div class="zchips">' + chips + ' <small>click: only this height · ctrl+click: add/remove</small></div>' +
    (order.length ? '' : '<p><small>No frame sees the part at these heights.</small></p>') +
    '<div class="imgs">' + order.map(f => { const x = byFrame[f], rd = r ? r.readings.find(y => y.frame === f) : null;
      const same = rd && readZ[f] === x.z, win = r && r.frame === f && same;
      return '<figure' + (win ? ' style="outline:2px solid #6cf"' : '') + '><a href="' + esc(BASE + x.file) + '" target="_blank"><img src="' + esc(BASE + x.file) + '"></a>' +
        '<figcaption>' + esc(f) + ' · z ' + x.z.toFixed(1) + ' · cover ' + x.cover.toFixed(2) +
        (rd ? '<br><span style="color:#6cf">' + esc(rd.lines.join(' / ')) + '</span> <small>' + rd.score.toFixed(2) + ', ' + rd.turn + '°' +
          (same ? '' : ', read at z ' + readZ[f].toFixed(1)) + '</small>' : r ? '<br><small>nothing read</small>' : '') +
        '</figcaption></figure>'; }).join('') + '</div>';
}
function bindViews() {
  const box = cb('viewsBlock');
  if (!box) return;
  box.querySelectorAll('button.zc').forEach(e => e.onclick = ev => {
    const z = e.dataset.z;
    if (z === 'all') viewZ = null;
    else if (ev.ctrlKey || ev.metaKey) {
      const zs = [...new Set((VIEWS[part.ref].all || VIEWS[part.ref].views).map(x => x.z))];
      viewZ = viewZ ? new Set(viewZ) : new Set(zs);
      viewZ.has(+z) ? viewZ.delete(+z) : viewZ.add(+z);
      if (!viewZ.size || viewZ.size === zs.length) viewZ = null;
    } else viewZ = new Set([+z]);
    box.innerHTML = viewsHtml(part.ref); bindViews();
  });
}

function focusPart(p) {
  part = p; picked = null; renderPart(); requestDraw();
  const pad = Math.max(p.box[2] - p.box[0], p.box[3] - p.box[1]) * 3 + 5 * cadPxPerMm;
  viewer.viewport.fitBounds(main().imageToViewportRectangle(p.cx - pad / 2, p.cy - pad / 2, pad, pad));
}

cb('cbCad').addEventListener('change', requestDraw);
cb('cbPads').addEventListener('change', requestDraw);
cb('cbFindings').addEventListener('change', requestDraw);
cb('cadMode').addEventListener('change', () => { cadHidden.clear(); cadChanged(); });
cb('cadFind').addEventListener('keydown', e => {
  if (e.key !== 'Enter' || !CAD) return;
  const q = e.target.value.trim().toLowerCase(), n = q.replace(/\s/g, '');
  const p = CAD.parts.find(p => p.ref.toLowerCase() === q) || CAD.parts.find(p => p.line && (p.line.name.toLowerCase() === q ||
    (p.line.value && p.line.value.text.toLowerCase().replace(/\s/g, '') === n)));
  if (p) focusPart(p);
});

// ---------------------------------------------------------------- parts list: every component, upright crop, counts
function lineValueKey(l) { return l && l.value ? l.value.value : Infinity; }
function renderList() {
  const b = CAD.board, g = cb('lGroup').value, fit = cb('lFit').value, q = cb('lFind').value.trim().toLowerCase();
  const side = cb('lSide').value, chk = cb('lCheck').value;
  const qn = q.replace(/\s/g, '');
  const groups = {};
  let shown = 0;
  for (const [ref, c] of Object.entries(b.components)) {
    const l = lineOf(c), grp = groupOf(l);
    if (g && grp !== g) continue;
    if (fit && c.fit !== fit) continue;
    if (side && c.side !== side) continue;
    const mk = markOf(ref), st = mk && mk.check ? mk.check.status : '';
    if (chk === 'problems' && !PROBLEM.has(st)) continue;
    if ((chk === 'ok' || chk === 'unread') && st !== chk) continue;
    if (chk === 'text' && !(mk && mk.lines.length)) continue;
    const oa = mk && mk.ocr ? (mk.ocr.otherTurn ? mk.ocr.otherTurn.agree : mk.ocr.agree) : '';
    if (chk === 'ocrdiff' && !(oa === 'differs' || oa === 'ocr-only')) continue;
    if (chk === 'ocrsame' && !(oa === 'same' || oa === 'close' || oa === 'partial')) continue;
    if (chk === 'vlmdiff' && !(mk && mk.vlm && (mk.vlm.agree === 'differs' || mk.vlm.agree === 'ocr-only'))) continue;
    if (chk === 'viewsdiff' && !(mk && mk.vlmViews && (mk.vlmViews.agree === 'differs' || mk.vlmViews.agree === 'ocr-only'))) continue;
    if (chk === 'codediff' && !(mk && mk.resistorCode && mk.resistorCode.agree !== 'same')) continue;
    const mtext = mk ? (mk.lines.join(' ') + ' ' + (mk.ocr ? mk.ocr.lines.join(' ') : '') + ' ' +
      (mk.vlm ? mk.vlm.lines.join(' ') : '') + ' ' + (mk.vlmViews ? mk.vlmViews.lines.join(' ') : '')).toLowerCase() : '';
    if (q && !mtext.includes(q) && !(ref.toLowerCase().includes(q) || (l && (l.name.toLowerCase().includes(q) ||
        (l.value && l.value.text.toLowerCase().replace(/\s/g, '') === qn))))) continue;
    const key = c.line == null ? 'none:' + (c.package || '') : 'line:' + c.line;
    ((groups[grp] = groups[grp] || {})[key] = (groups[grp][key] || [])).push(ref);
    shown++;
  }
  cb('lCount').textContent = shown + ' of ' + Object.keys(b.components).length;
  const card = ref => {
    const c = b.components[ref], cr = CAD.crops[ref];
    const img = cr ? '<img loading="lazy" style="aspect-ratio:' + cr.w + '/' + cr.h + '" src="' + esc(BASE + cr.file) + '" title="' + (cr.w / cr.pxPerMm).toFixed(1) + ' × ' +
      (cr.h / cr.pxPerMm).toFixed(1) + ' mm">' : '<div class="noimg">' + esc(c.side) + ' side</div>';
    const mk = markOf(ref), st = mk && mk.check ? mk.check.status : '';
    const border = PROBLEM.has(st) ? CHECK_COL[st] : (c.fit === 'fitted' ? '#333' : FIT[c.fit]);
    const mkHtml = mk ? '<div class="mk">' + (mk.lines.length ? mk.lines.map(esc).join('<br>') :
      '<span style="color:#666">' + (mk.present === false ? 'empty' : '—') + '</span>') + '</div>' +
      (st && st !== 'info' ? '<div class="ck" style="color:' + CHECK_COL[st] + '">' + esc(st) +
        (PROBLEM.has(st) || st === 'unread' ? ': ' + esc(mk.check.text) : '') + '</div>' : '') +
      (mk.ocr && (mk.lines.length || mk.ocr.lines.length) ? '<div class="oc">' + ocrBadge(mk) + '</div>' : '') +
      (mk.vlm && (mk.lines.length || mk.vlm.lines.length) ? '<div class="oc">' + vlmBadge(mk) + '</div>' : '') +
      (mk.vlmViews && (mk.lines.length || mk.vlmViews.lines.length) ? '<div class="oc">' + viewsBadge(mk) + '</div>' : '') +
      (mk.resistorCode ? '<div class="oc">' + codeBadge(mk) + '</div>' : '') : '';
    return '<div class="lcard" data-ref="' + esc(ref) + '" style="border-color:' + border + '" title="' +
      esc(mk ? (mk.note || '') + (mk.conf ? ' · read ' + mk.conf : '') : '') + '">' + img + '<div class="r">' + esc(ref) + '</div>' +
      mkHtml + '</div>';
  };
  let h = '';
  for (const grp of GROUP_ORDER.filter(x => groups[x]).concat(Object.keys(groups).filter(x => !GROUP_ORDER.includes(x)))) {
    const lines = Object.entries(groups[grp]);
    const refs = lines.flatMap(([, r]) => r), byFit = {};
    for (const r of refs) byFit[b.components[r].fit] = (byFit[b.components[r].fit] || 0) + 1;
    h += '<h2>' + esc(grp) + ' <small>' + refs.length + ' · ' + Object.entries(byFit).map(([k, n]) => k + ' ' + n).join(', ') +
      ' · ' + lines.length + ' kinds</small></h2>';
    lines.sort(([ka, ra], [kb, rb]) => {
      const la = ka.startsWith('line:') ? b.lines[+ka.slice(5)] : null, lb = kb.startsWith('line:') ? b.lines[+kb.slice(5)] : null;
      return (lineValueKey(la) - lineValueKey(lb)) || natKey(la ? la.name : ka).localeCompare(natKey(lb ? lb.name : kb));
    });
    for (const [key, rs] of lines) {
      rs.sort((x, y) => natKey(x).localeCompare(natKey(y)));
      const l = key.startsWith('line:') ? b.lines[+key.slice(5)] : null;
      let head;
      if (l) {
        const codes = l.value && l.value.marking ? Object.entries(l.value.marking).map(([k, v]) =>
          '<code title="' + ({ zero: '0 Ω', 3: '3-digit', 4: '4-digit', eia96: 'EIA-96' })[k] + '">' + esc(v) + '</code>').join('') : '';
        head = '<b>' + esc(l.value ? l.value.text : l.name) + '</b>' + (l.value && l.value.voltage ? ' ' + l.value.voltage + ' V' : '') +
          (l.value && l.name !== l.value.text ? ' <small>(' + esc(l.name) + ')</small>' : '') +
          ' <small>' + esc(l.footprint) + ' · ×' + rs.length + (l.fit !== 'fitted' ? ' · <span style="color:' + FIT[l.fit] + '">' +
          esc(FIT_LABEL[l.fit]) + '</span>' : '') + '</small>' + (codes ? ' <small>marking may read</small>' + codes : '');
      } else {
        head = '<b>' + esc(key.slice(5) || 'no package') + '</b> <small>not in BOM · ×' + rs.length + '</small>';
      }
      h += '<div class="lline">' + head + '</div><div class="lcards">' + rs.map(card).join('') + '</div>';
    }
  }
  cb('lBody').innerHTML = h || '<p><small>nothing matches</small></p>';
  cb('lBody').querySelectorAll('.lcard').forEach(e => e.onclick = () => {
    const p = CAD.byRef[e.dataset.ref];
    if (!p) return;   // other side: not on this panorama
    cb('list').style.display = 'none'; focusPart(p);
  });
}
function openList() {
  if (!CAD) return;
  const counts = {};
  for (const c of Object.values(CAD.board.components)) { const g = groupOf(lineOf(c)); counts[g] = (counts[g] || 0) + 1; }
  if (!cb('lGroup').options.length) cb('lSide').value = CAD.board.align.side;   // the side this panorama shows
  if (cb('lGroup').options.length < 2)
    cb('lGroup').innerHTML = '<option value="">all groups</option>' + GROUP_ORDER.filter(g => counts[g]).map(g =>
      '<option value="' + esc(g) + '">' + esc(g) + ' (' + counts[g] + ')</option>').join('');
  cb('list').style.display = 'block';
  renderList();
}
cb('btnList').addEventListener('click', openList);
cb('lClose').addEventListener('click', () => { cb('list').style.display = 'none'; });
for (const id of ['lGroup', 'lFit', 'lSide', 'lCheck']) cb(id).addEventListener('change', renderList);
cb('lFind').addEventListener('input', renderList);
cb('lSize').addEventListener('input', () => cb('lBody').style.setProperty('--h', cb('lSize').value + 'px'));
cb('lBody').style.setProperty('--h', cb('lSize').value + 'px');
document.addEventListener('keydown', e => {
  if (e.key === 'Escape' && cb('list').style.display === 'block') cb('list').style.display = 'none';
});

// ---------------------------------------------------------------- diagnostics HUD: where the time goes
// Per second: frames (rAF), overlay draw time, OSD tiles drawn / loaded, tile download time (Resource Timing),
// long main-thread tasks. "copy" puts the last 60 s as JSON on the clipboard to paste into a bug report.
const HUD = { on: false, frames: [], draws: [], tilesDrawn: 0, tilesLoaded: 0, tileMs: [], longTasks: [], anim: 0,
              last: 0, history: [], cadStats: null };
viewer.addHandler('tile-drawn', () => { if (HUD.on) HUD.tilesDrawn++; });
viewer.addHandler('tile-loaded', () => { if (HUD.on) HUD.tilesLoaded++; });
viewer.addHandler('animation', () => { if (HUD.on) HUD.anim++; });
try { new PerformanceObserver(l => { if (HUD.on) for (const e of l.getEntries()) if (e.name.includes('_files/')) HUD.tileMs.push(e.duration); })
  .observe({ type: 'resource' }); } catch (e) {}
try { new PerformanceObserver(l => { if (HUD.on) for (const e of l.getEntries()) HUD.longTasks.push(e.duration); })
  .observe({ type: 'longtask' }); } catch (e) {}
const pct = (a, q) => { if (!a.length) return 0; const b = [...a].sort((x, y) => x - y); return b[Math.min(b.length - 1, Math.floor(b.length * q))]; };
function hudLoop(t) { if (!HUD.on) return; if (HUD.last) HUD.frames.push(t - HUD.last); HUD.last = t; requestAnimationFrame(hudLoop); }
function hudTick() {
  if (!HUD.on) return;
  const fr = HUD.frames, it = main();
  const snap = {
    t: Math.round(performance.now() / 1000),
    fps: fr.length, frameMedMs: +pct(fr, .5).toFixed(1), frameP95Ms: +pct(fr, .95).toFixed(1), frameMaxMs: +Math.max(0, ...fr).toFixed(1),
    overlayDraws: HUD.draws.length, overlayMsAvg: +(HUD.draws.reduce((a, b) => a + b, 0) / (HUD.draws.length || 1)).toFixed(2),
    overlayMsMax: +Math.max(0, ...HUD.draws).toFixed(2), animEvents: HUD.anim,
    tilesDrawn: HUD.tilesDrawn, tilesPerFrame: +(HUD.tilesDrawn / (fr.length || 1)).toFixed(1), tilesLoaded: HUD.tilesLoaded,
    tileFetchMedMs: +pct(HUD.tileMs, .5).toFixed(0), tileFetchMaxMs: +Math.max(0, ...HUD.tileMs).toFixed(0),
    longTasks: HUD.longTasks.length, longTaskMaxMs: +Math.max(0, ...HUD.longTasks).toFixed(0),
    zoomImgPxPerScreenPx: it ? +(1 / it.viewportToImageZoom(viewer.viewport.getZoom(true))).toFixed(3) : null,
    tilesCached: viewer.tileCache ? viewer.tileCache.numTilesLoaded() : null,
    heapMB: performance.memory ? Math.round(performance.memory.usedJSHeapSize / 1e6) : null,
  };
  HUD.history.push(snap); if (HUD.history.length > 60) HUD.history.shift();
  HUD.frames = []; HUD.draws = []; HUD.tilesDrawn = 0; HUD.tilesLoaded = 0; HUD.tileMs = []; HUD.longTasks = []; HUD.anim = 0;
  let hint = 'idle';
  if (snap.animEvents) {
    if (snap.longTasks && snap.longTaskMaxMs > 50) hint = 'main thread blocked (long tasks) — JS or canvas work';
    else if (snap.overlayMsMax > 8) hint = 'overlay drawing is slow';
    else if (snap.frameP95Ms > 40 && snap.tilesPerFrame > 40) hint = 'many tiles per frame — tile drawing / GPU';
    else if (snap.frameP95Ms > 40) hint = 'slow frames, not the overlay — browser compositing / GPU';
    else if (snap.tileFetchMedMs > 150) hint = 'tiles arrive slowly — server / disk';
    else hint = 'ok';
  }
  cb('hud').innerHTML = esc(
    'fps ' + snap.fps + '   frame ' + snap.frameMedMs + ' / p95 ' + snap.frameP95Ms + ' / max ' + snap.frameMaxMs + ' ms\n' +
    'overlay ' + snap.overlayDraws + ' draws, avg ' + snap.overlayMsAvg + ' max ' + snap.overlayMsMax + ' ms\n' +
    'tiles drawn ' + snap.tilesDrawn + ' (' + snap.tilesPerFrame + '/frame), loaded ' + snap.tilesLoaded +
      ', fetch ' + snap.tileFetchMedMs + ' / max ' + snap.tileFetchMaxMs + ' ms, cached ' + snap.tilesCached + '\n' +
    'long tasks ' + snap.longTasks + ' (max ' + snap.longTaskMaxMs + ' ms)   zoom ' + snap.zoomImgPxPerScreenPx + ' img px/screen px' +
      (snap.heapMB ? '   heap ' + snap.heapMB + ' MB' : '') + '\n' +
    'dpr ' + (window.devicePixelRatio || 1) + '   canvas ' + canvas.width + '×' + canvas.height + '   → ' + hint + '\n') +
    '<button id="hudCopy">copy last 60 s</button>';
  cb('hudCopy').onclick = () => navigator.clipboard.writeText(JSON.stringify({
    userAgent: navigator.userAgent, dpr: window.devicePixelRatio, screen: [innerWidth, innerHeight],
    image: [DATA.width, DATA.height], cad: HUD.cadStats, history: HUD.history }, null, 1));
}
function setHud(on) {
  HUD.on = on; cb('cbHud').checked = on; cb('hud').style.display = on ? 'block' : 'none';
  if (on) { HUD.last = 0; requestAnimationFrame(hudLoop); cb('hud').textContent = 'measuring… zoom / pan to see the numbers'; }
}
setInterval(hudTick, 1000);
cb('cbHud').addEventListener('change', () => setHud(cb('cbHud').checked));
document.addEventListener('keydown', e => {
  if ((e.key === 'd' || e.key === 'D') && !/INPUT|SELECT|TEXTAREA/.test(document.activeElement.tagName)) setHud(!HUD.on);
});

// ---------------------------------------------------------------- one overlay redraw per frame
let drawQueued = false;
function requestDraw() {
  if (drawQueued) return;
  drawQueued = true;
  const run = () => { if (!drawQueued) return; drawQueued = false; draw(); };
  requestAnimationFrame(run);
  setTimeout(run, 100);   // rAF does not fire in a hidden tab
}
const drawImpl = draw;
draw = function () { const t0 = performance.now(); drawImpl(); if (HUD.on) HUD.draws.push(performance.now() - t0); };

if (!DATA.regions.length) cb('lblRegions').style.display = 'none';
for (const id of ['cbFrames', 'cbPairs', 'cbRegions']) cb(id).addEventListener('change', () => {
  cb('legend').style.display = cb('cbPairs').checked ? 'block' : 'none'; draw(); });
cb('cbSeams').addEventListener('change', () => setSeams(cb('cbSeams').checked));
for (const ev of ['open', 'animation', 'update-viewport', 'resize']) viewer.addHandler(ev, requestDraw);
window.addEventListener('resize', requestDraw);
</script>
</body>
</html>
'''

TEMPLATE_3D = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PCB Relief 3D</title>
<style>
  html, body { margin: 0; height: 100%; background: #111; color: #ddd; font: 13px system-ui, sans-serif; overflow: hidden; }
  #panel { position: absolute; top: 8px; left: 8px; background: rgba(20,20,20,.85); padding: 8px 10px;
           border-radius: 6px; z-index: 10; max-width: 320px; }
  #panel label { display: block; margin: 3px 0; cursor: pointer; }
  #panel input[type=range] { width: 140px; vertical-align: middle; }
  #status { position: absolute; bottom: 8px; left: 8px; background: rgba(20,20,20,.85); padding: 4px 8px;
            border-radius: 4px; }
  small { color: #999; }
  a { color: #6cf; }
</style>
<script type="importmap">
{ "imports": {
  "three": "https://cdn.jsdelivr.net/npm/three@__THREE__/build/three.module.js",
  "three/addons/": "https://cdn.jsdelivr.net/npm/three@__THREE__/examples/jsm/"
} }
</script>
</head>
<body>
<div id="panel">
  <b>PCB relief</b> <a href="index.html" style="margin-left:8px">&larr; 2D viewer</a><br>
  <small>height from focus, all-in-focus texture</small>
  <label>vertical scale <input type="range" id="exag" min="1" max="8" step="0.5" value="2"> <span id="exagV"></span></label>
  <label><input type="radio" name="tex" value="photo" checked> photo</label>
  <label><input type="radio" name="tex" value="height"> height colours 0 &hellip;
    <input type="range" id="hRange" min="0.3" max="5" step="0.1" value="1.5" style="width:90px"> <span id="hRangeV"></span> mm</label>
  <div id="cadPanel" style="display:none;border-top:1px solid #444;margin-top:4px;padding-top:4px">
    <small>CAD reference (STEP top view, <a href="cad/summary.json">summary</a>)</small>
    <label><input type="radio" name="tex" value="cad"> CAD height colours (same range)</label>
    <label><input type="radio" name="tex" value="error"> error measured &minus; CAD &plusmn;
      <input type="range" id="eRange" min="0.2" max="2" step="0.1" value="0.6" style="width:80px"> <span id="eRangeV"></span> mm</label>
    <label><input type="checkbox" id="cadSurface"> CAD surface instead of measured</label>
    <label><input type="checkbox" id="cadOverlay"> CAD overlay (translucent)</label>
    <small id="cadInfo"></small>
  </div>
  <label><input type="checkbox" id="light" checked> relief shading</label>
  <label><input type="checkbox" id="hd"> high-res texture (8k)</label>
  <small>drag: rotate &middot; right-drag: pan &middot; wheel: zoom &middot; double-click: orbit around point &middot; click: read heights</small>
</div>
<div id="status">loading&hellip;</div>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const D = __DATA__;
const status = document.getElementById('status');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(innerWidth, innerHeight);
renderer.colorSpace = THREE.SRGBColorSpace;
document.body.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111111);
const mm = D.mmPerPx;                               // per panorama px
const Wmm = D.width * mm, Hmm = D.height * mm;
const camera = new THREE.PerspectiveCamera(40, innerWidth / innerHeight, 0.5, 5000);
camera.up.set(0, 0, 1);
camera.position.set(0, -Hmm * 0.9, Hmm * 0.9);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.screenSpacePanning = true;
controls.maxPolarAngle = Math.PI * 0.49;

scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 1.4);
sun.position.set(-60, -40, 120);
scene.add(sun);

const [cw, ch] = D.cells, step = D.meshStep;
const nx = Math.floor((cw - 1) / step) + 1, ny = Math.floor((ch - 1) / step) + 1;
const heights = new Int16Array(await (await fetch('mesh.bin')).arrayBuffer());   // um, -32768 = no data
let surface = heights;                              // geometry source: measured (mesh.bin) or CAD (cad_height.bin)
const hAt = (i, j) => heights[(j * step) * cw + i * step];
const sAt = (i, j) => { const v = surface[(j * step) * cw + i * step]; return v === -32768 ? 0 : v; };
const geo = new THREE.BufferGeometry();
const pos = new Float32Array(nx * ny * 3), uv = new Float32Array(nx * ny * 2);
for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
  const n = j * nx + i, px = (i * step + 0.5) * D.cell, py = (j * step + 0.5) * D.cell;
  pos[3 * n] = px * mm - Wmm / 2; pos[3 * n + 1] = Hmm / 2 - py * mm;
  uv[2 * n] = px / D.width; uv[2 * n + 1] = 1 - py / D.height;
}
const idx = new Uint32Array((nx - 1) * (ny - 1) * 6);
let t = 0;
for (let j = 0; j < ny - 1; j++) for (let i = 0; i < nx - 1; i++) {
  if (hAt(i, j) === -32768 || hAt(i + 1, j) === -32768 || hAt(i, j + 1) === -32768 || hAt(i + 1, j + 1) === -32768)
    continue;
  const a = j * nx + i, b = a + 1, c = a + nx, d = c + 1;
  idx.set([a, c, b, b, c, d], t); t += 6;
}
geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
geo.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
geo.setIndex(new THREE.BufferAttribute(idx.subarray(0, t), 1));
const steep = new Float32Array(nx * ny);
geo.setAttribute('steep', new THREE.BufferAttribute(steep, 1));
// wall colour: uv of the highest vertex within 3 vertices (the top edge above the wall)
const wallUv = new Float32Array(nx * ny * 2);
for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
  let best = -1e9, bi = i, bj = j;
  for (let dj = -3; dj <= 3; dj++) for (let di = -3; di <= 3; di++) {
    const ii = i + di, jj = j + dj;
    if (ii < 0 || jj < 0 || ii >= nx || jj >= ny) continue;
    const hv = hAt(ii, jj);
    if (hv > best) { best = hv; bi = ii; bj = jj; }
  }
  const n = j * nx + i, m = bj * nx + bi;
  wallUv[2 * n] = uv[2 * m]; wallUv[2 * n + 1] = uv[2 * m + 1];
}
geo.setAttribute('wallUv', new THREE.BufferAttribute(wallUv, 2));
function setExaggeration(k) {
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++)
    pos[3 * (j * nx + i) + 2] = Math.max(-2, (surface === heights ? hAt(i, j) : sAt(i, j)) / 1000) * k;
  geo.attributes.position.needsUpdate = true;
  // normals from a 5x5 box-blurred surface (staircase walls would show stripes)
  const zb = new Float32Array(nx * ny), tmp = new Float32Array(nx * ny), r = 2;
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    let a = 0, c = 0;
    for (let d = -r; d <= r; d++) { const ii = i + d; if (ii >= 0 && ii < nx) { a += pos[3 * (j * nx + ii) + 2]; c++; } }
    tmp[j * nx + i] = a / c;
  }
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    let a = 0, c = 0;
    for (let d = -r; d <= r; d++) { const jj = j + d; if (jj >= 0 && jj < ny) { a += tmp[jj * nx + i]; c++; } }
    zb[j * nx + i] = a / c;
  }
  const nrm = new Float32Array(nx * ny * 3), sp = step * D.cell * mm;
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const n = j * nx + i;
    const dx = (zb[j * nx + Math.min(nx - 1, i + 1)] - zb[j * nx + Math.max(0, i - 1)]) / (2 * sp);
    const dy = (zb[Math.max(0, j - 1) * nx + i] - zb[Math.min(ny - 1, j + 1) * nx + i]) / (2 * sp);   // +y is up the image
    const l = Math.hypot(dx, dy, 1);
    nrm[3 * n] = -dx / l; nrm[3 * n + 1] = -dy / l; nrm[3 * n + 2] = 1 / l;
    // steepness: 0 on tops and the board, 1 on walls (normal z below 0.4)
    steep[n] = Math.min(1, Math.max(0, (0.8 - nrm[3 * n + 2]) / 0.4));
  }
  geo.setAttribute('normal', new THREE.BufferAttribute(nrm, 3));
  geo.attributes.steep.needsUpdate = true;
  geo.computeBoundingSphere();
  document.getElementById('exagV').textContent = k + 'x';
  cadOverlayUpdate(k);
}
// ---- CAD reference (compareCad.py writes cad/viewer.json + int16 um rasters on the same cells)
let cad = null, overlay = null;
function cadOverlayUpdate(k) {
  if (!overlay) return;
  const p = overlay.geometry.attributes.position.array;
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const v = cad.height[(j * step) * cw + i * step];
    p[3 * (j * nx + i) + 2] = (v === -32768 ? 0 : v / 1000) * k + 0.01;
  }
  overlay.geometry.attributes.position.needsUpdate = true;
  overlay.geometry.computeVertexNormals();
  overlay.geometry.computeBoundingSphere();
}
setExaggeration(+document.getElementById('exag').value);

const loader = new THREE.TextureLoader();
const textures = {};
function texture(url) {
  if (!textures[url]) {
    textures[url] = loader.load(url, () => { status.textContent = url + ' loaded'; });
    textures[url].colorSpace = THREE.SRGBColorSpace;
    textures[url].anisotropy = renderer.capabilities.getMaxAnisotropy();
  }
  return textures[url];
}
// steep faces read the texture at the top edge above (wallUv) from a blurred mip level
function wallShading(material) {
  material.onBeforeCompile = shader => {
    shader.vertexShader = `attribute float steep;
attribute vec2 wallUv;
varying float vSteep;
varying vec2 vWallUv;
` + shader.vertexShader.replace('#include <uv_vertex>', `#include <uv_vertex>
  vSteep = steep;
  vWallUv = wallUv;`);
    shader.fragmentShader = `varying float vSteep;
varying vec2 vWallUv;
` + shader.fragmentShader.replace('#include <map_fragment>', `#ifdef USE_MAP
  vec4 sampledDiffuseColor = mix(texture2D(map, vMapUv), texture2D(map, vWallUv, 3.0), vSteep);
  diffuseColor *= sampledDiffuseColor;
#endif`);
  };
  return material;
}
const lit = wallShading(new THREE.MeshLambertMaterial({ map: texture('focus_tex_4k.jpg') }));
const flat = wallShading(new THREE.MeshBasicMaterial({ map: lit.map }));
const mesh = new THREE.Mesh(geo, lit);
scene.add(mesh);
status.textContent = nx + ' x ' + ny + ' vertices, ' + Wmm.toFixed(0) + ' x ' + Hmm.toFixed(0) + ' mm, camera ' +
  D.cameraDistanceMm + ' mm, ortho passes ' + D.orthoPasses;

function turbo(x) {
  const r = 0.13572138 + x * (4.61539260 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943))));
  const g = 0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604))));
  const b = 0.10667330 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973))));
  return [r, g, b].map(v => Math.max(0, Math.min(255, Math.round(v * 255))));
}
let heightTex = null, heightTexRange = null;
function heightTexture(vmax) {     // turbo 0..vmax from the mesh heights, rows flipped (texture v = 0 at the bottom)
  if (heightTex && heightTexRange === vmax) return heightTex;
  const data = new Uint8Array(cw * ch * 4), lut = [];
  for (let i = 0; i < 256; i++) lut.push(turbo(i / 255));
  for (let r = 0; r < ch; r++) for (let c = 0; c < cw; c++) {
    const v = heights[r * cw + c], n = (ch - 1 - r) * cw + c;
    const t = v === -32768 ? [40, 40, 40] : lut[Math.max(0, Math.min(255, Math.round(v / 1000 / vmax * 255)))];
    data.set([t[0], t[1], t[2], 255], 4 * n);
  }
  if (heightTex) heightTex.dispose();
  heightTex = new THREE.DataTexture(data, cw, ch, THREE.RGBAFormat);
  heightTex.colorSpace = THREE.SRGBColorSpace;
  heightTex.magFilter = THREE.LinearFilter;
  heightTex.needsUpdate = true;
  heightTexRange = vmax;
  return heightTex;
}
const dataTex = {};
function rasterTexture(key, fn) {   // RGBA texture over all cells, rows flipped; fn(cell index) -> [r, g, b]
  if (dataTex[key]) return dataTex[key];
  for (const k in dataTex) { dataTex[k].dispose(); delete dataTex[k]; }
  const data = new Uint8Array(cw * ch * 4);
  for (let r = 0; r < ch; r++) for (let c = 0; c < cw; c++) {
    const t = fn(r * cw + c);
    data.set([t[0], t[1], t[2], 255], 4 * ((ch - 1 - r) * cw + c));
  }
  const tx = new THREE.DataTexture(data, cw, ch, THREE.RGBAFormat);
  tx.colorSpace = THREE.SRGBColorSpace; tx.magFilter = THREE.LinearFilter; tx.needsUpdate = true;
  return dataTex[key] = tx;
}
function diverging(x) {   // -1 blue .. 0 white .. 1 red
  const P = [[5, 48, 97], [67, 147, 195], [247, 247, 247], [214, 96, 77], [103, 0, 31]];
  const f = (Math.max(-1, Math.min(1, x)) + 1) * 2, i = Math.min(3, Math.floor(f)), u = f - i;
  return P[i].map((v, n) => Math.round(v + (P[i + 1][n] - v) * u));
}
function cadTexture(vmax) {
  const lut = []; for (let i = 0; i < 256; i++) lut.push(turbo(i / 255));
  return rasterTexture('cad' + vmax, n => {
    const v = cad.height[n];
    return v === -32768 ? [40, 40, 40] : lut[Math.max(0, Math.min(255, Math.round(v / 1000 / vmax * 255)))];
  });
}
function errorTexture(span) {
  return rasterTexture('err' + span, n => {
    const v = cad.error[n];
    if (v === -32768) return [25, 25, 25];
    const t = diverging(v / 1000 / span);
    return cad.state[n] === 2 ? t : t.map(x => Math.round(x * 0.25 + 60));   // dim: filled or above the range
  });
}
function buildOverlay() {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(pos), 3));
  const ix = new Uint32Array((nx - 1) * (ny - 1) * 6);
  let m = 0;
  const off = (i, j) => cad.height[(j * step) * cw + i * step] === -32768;
  for (let j = 0; j < ny - 1; j++) for (let i = 0; i < nx - 1; i++) {
    if (off(i, j) || off(i + 1, j) || off(i, j + 1) || off(i + 1, j + 1)) continue;
    const a = j * nx + i, b = a + 1, c = a + nx, d = c + 1;
    ix.set([a, c, b, b, c, d], m); m += 6;
  }
  g.setIndex(new THREE.BufferAttribute(ix.subarray(0, m), 1));
  overlay = new THREE.Mesh(g, new THREE.MeshLambertMaterial({ color: 0x66ccff, transparent: true, opacity: 0.45,
    depthWrite: false, side: THREE.DoubleSide }));
  overlay.renderOrder = 1;
  scene.add(overlay);
  cadOverlayUpdate(+document.getElementById('exag').value);
}
async function loadCad() {
  const r = await fetch('cad/viewer.json').catch(() => null);
  if (!r || !r.ok) return;
  const info = await r.json();
  if (info.cells[0] !== cw || info.cells[1] !== ch) { console.warn('cad/viewer.json: other cell grid'); return; }
  const bin = async (u, T) => new T(await (await fetch(u)).arrayBuffer());
  cad = { info, height: await bin(info.files.cad, Int16Array), error: await bin(info.files.error, Int16Array),
          state: await bin(info.files.state, Uint8Array) };
  document.getElementById('cadPanel').style.display = '';
  document.getElementById('cadInfo').textContent = 'XY shift ' + info.shiftCells.join(', ') + ' cells, fit scale ' +
    info.fit.scale + ', bias ' + info.fit.bias + ' mm, stack range top ~' + info.rangeTop + ' mm';
  document.getElementById('eRangeV').textContent = (+document.getElementById('eRange').value).toFixed(1);
}
function updateMaterial() {
  const tex = document.querySelector('input[name=tex]:checked').value;
  const hd = document.getElementById('hd').checked && renderer.capabilities.maxTextureSize >= 8192;
  const vmax = +document.getElementById('hRange').value;
  document.getElementById('hRangeV').textContent = vmax.toFixed(1);
  const map = tex === 'height' ? heightTexture(vmax)
    : tex === 'cad' && cad ? cadTexture(vmax)
    : tex === 'error' && cad ? errorTexture(+document.getElementById('eRange').value)
    : texture(hd ? 'focus_tex.jpg' : 'focus_tex_4k.jpg');
  lit.map = flat.map = map; lit.needsUpdate = flat.needsUpdate = true;
  mesh.material = document.getElementById('light').checked ? lit : flat;
}
document.querySelectorAll('input[name=tex], #light, #hd, #hRange, #eRange').forEach(e => e.addEventListener('change', updateMaterial));
document.getElementById('eRange').addEventListener('input', e => {
  document.getElementById('eRangeV').textContent = (+e.target.value).toFixed(1); });
document.getElementById('cadSurface').addEventListener('change', e => {
  surface = e.target.checked && cad ? cad.height : heights;
  setExaggeration(+document.getElementById('exag').value);
});
document.getElementById('cadOverlay').addEventListener('change', e => {
  if (e.target.checked && !overlay && cad) buildOverlay();
  if (overlay) overlay.visible = e.target.checked;
});
loadCad();
document.getElementById('hRange').addEventListener('input', e => {
  document.getElementById('hRangeV').textContent = (+e.target.value).toFixed(1); });
document.getElementById('hRangeV').textContent = '1.5';
document.getElementById('exag').addEventListener('input', e => setExaggeration(+e.target.value));

const ray = new THREE.Raycaster();
renderer.domElement.addEventListener('dblclick', e => {
  ray.setFromCamera(new THREE.Vector2(e.clientX / innerWidth * 2 - 1, -e.clientY / innerHeight * 2 + 1), camera);
  const hit = ray.intersectObject(mesh)[0];
  if (hit) { camera.position.add(hit.point.clone().sub(controls.target)); controls.target.copy(hit.point); }
});
let down = null;
renderer.domElement.addEventListener('pointerdown', e => { down = [e.clientX, e.clientY]; });
renderer.domElement.addEventListener('click', e => {   // readout: measured / CAD / error at the clicked cell
  if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) return;
  ray.setFromCamera(new THREE.Vector2(e.clientX / innerWidth * 2 - 1, -e.clientY / innerHeight * 2 + 1), camera);
  const hit = ray.intersectObject(mesh)[0];
  if (!hit) return;
  const c = Math.floor((hit.point.x + Wmm / 2) / mm / D.cell), r = Math.floor((Hmm / 2 - hit.point.y) / mm / D.cell);
  if (c < 0 || r < 0 || c >= cw || r >= ch) return;
  const n = r * cw + c, f = v => v === -32768 ? '-' : (v / 1000).toFixed(2);
  let t = 'cell ' + c + ', ' + r + ': measured ' + f(heights[n]) + ' mm';
  if (cad) t += ', CAD ' + f(cad.height[n]) + ', error ' + f(cad.error[n]) +
    (cad.state[n] === 2 ? '' : cad.state[n] === 1 ? ' (filled / above range)' : '');
  status.textContent = t;
});
addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});
window.relief = { camera, controls, mesh, mm, Wmm, Hmm, cad: () => cad };   // for debugging from the console
renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });
</script>
</body>
</html>
'''


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Re-render index.html of stitchGrid / stackFocus output folders with the '
                                             'current viewer template (no re-stitching).')
    ap.add_argument('folders', nargs='+', help='output folders, e.g. out/captured_51.5')
    ap.add_argument('--board', help='panorama folder with board.json + markings to show on these pages, e.g. a stack '
                                    'page: BuildViewer.py out/stack --board out/captured_51.5 (serve out/)')
    a = ap.parse_args()
    for folder in a.folders:
        rebuildHtml(folder, a.board)
        print(f'{folder}/index.html rebuilt')
