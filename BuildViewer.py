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
</div>
<div id="info"></div>
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
function showLayer(i) {             // -1 = all-in-focus (main item), 'm' = marking, else index of a height
  layerIdx = i;
  for (const k in layerItems) layerItems[k].setOpacity(k === String(i) ? 1 : 0);
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
  document.querySelectorAll('input[name=layer]').forEach(r => r.addEventListener('change', () =>
    showLayer(r.value === '-1' ? -1 : r.value === 'm' ? 'm' : +cb('zSlider').value)));
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

if (!DATA.regions.length) cb('lblRegions').style.display = 'none';
for (const id of ['cbFrames', 'cbPairs', 'cbRegions']) cb(id).addEventListener('change', () => {
  cb('legend').style.display = cb('cbPairs').checked ? 'block' : 'none'; draw(); });
cb('cbSeams').addEventListener('change', () => setSeams(cb('cbSeams').checked));
for (const ev of ['open', 'animation', 'update-viewport', 'resize']) viewer.addHandler(ev, draw);
window.addEventListener('resize', draw);
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
  <label><input type="checkbox" id="light" checked> relief shading</label>
  <label><input type="checkbox" id="hd"> high-res texture (8k)</label>
  <small>drag: rotate &middot; right-drag: pan &middot; wheel: zoom &middot; double-click: orbit around point</small>
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
const hAt = (i, j) => heights[(j * step) * cw + i * step];
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
    pos[3 * (j * nx + i) + 2] = Math.max(-2, hAt(i, j) / 1000) * k;
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
function updateMaterial() {
  const tex = document.querySelector('input[name=tex]:checked').value;
  const hd = document.getElementById('hd').checked && renderer.capabilities.maxTextureSize >= 8192;
  document.getElementById('hRangeV').textContent = (+document.getElementById('hRange').value).toFixed(1);
  const map = tex === 'height' ? heightTexture(+document.getElementById('hRange').value)
    : texture(hd ? 'focus_tex.jpg' : 'focus_tex_4k.jpg');
  lit.map = flat.map = map; lit.needsUpdate = flat.needsUpdate = true;
  mesh.material = document.getElementById('light').checked ? lit : flat;
}
document.querySelectorAll('input[name=tex], #light, #hd, #hRange').forEach(e => e.addEventListener('change', updateMaterial));
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
addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});
window.relief = { camera, controls, mesh, mm, Wmm, Hmm };   // for debugging from the console
renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });
</script>
</body>
</html>
'''
