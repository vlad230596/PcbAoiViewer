"""CLI: the board's STEP model with the stack's photo texture, for a browser viewer (CadTexture.py).

    uv run texturedModel.py out/stack --board boards/Antilatency.SocketTester.Main-r79 \
        [--align out/captured_51.5] [--image focus.jpg] [--tex-size 8192] [--no-parallax]
    uv run python -m http.server 8765 --directory out/stack     # then open /model.html

Needs boards/<board>/model/ (research/cad/venv/Scripts/python.exe stepToMesh.py boards/<board>), the board -> panorama
fit (alignBoard.py, its board.json; the stack must share the reference canvas, which a stack built from the same
reference stitch does) and the stack (focus.jpg, height.npy, index.html for frame positions, report.json for D).
Writes into the stack folder: model.glb, model_tex.jpg (8k) + model_tex_4k.jpg, model_parts.json,
model_relief.bin, model_registration.json, model.html.
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

import BoardData
from CadTexture import (BoardTexture, CadModel, GlbWriter, PanoramaProjector, RegistrationCheck, TopVisibility,
                        bodyColours, readEmbeddedData, reliefGrid)

THREE_VERSION = '0.160.0'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stack', help='stackFocus.py output folder (focus.jpg, height.npy, index.html, report.json)')
    ap.add_argument('--board', required=True, help='boards/<name> folder with model/mesh.npz + parts.json')
    ap.add_argument('--align', default='out/captured_51.5', help='folder with the alignBoard.py board.json')
    ap.add_argument('--image', default='focus.jpg', help='stack image used as the texture')
    ap.add_argument('--tex-size', type=int, default=8192, help='long side of the board texture, px')
    ap.add_argument('--margin', type=float, default=1.0, help='mm around the board outline in the texture')
    ap.add_argument('--no-parallax', action='store_true', help='uv = M x, y at every height (no correction)')
    ap.add_argument('--relief-step', type=float, default=0.2, help='mm grid of the measured relief overlay')
    ap.add_argument('--steep', type=float, default=0.5, help='default steep-face threshold (normal z) in the page')
    a = ap.parse_args()

    t0 = time.time()
    out = a.stack
    aligned = BoardData.loadBoard(a.align)
    M = np.array(aligned['align']['M'])
    pxPerMm = float(np.mean(aligned['align']['pxPerMm']))
    model = CadModel(os.path.join(a.board, 'model'), aligned['components'])
    print(f"model: {len(model.vertices)} vertices, {len(model.triangles)} triangles, {len(model.refs)} parts")
    missing = [r for r, c in aligned['components'].items() if r not in model.parts]
    if missing:
        print(f"in board.json but not in the STEP: {', '.join(missing)}")

    rep = json.load(open(os.path.join(out, 'report.json'), encoding='utf-8'))
    D = rep['stack']['cameraDistanceMm']
    data = readEmbeddedData(os.path.join(out, 'index.html'))
    centres = [(f['px'] + (data['frameW'] - 1) / 2, f['py'] + (data['frameH'] - 1) / 2) for f in data['frames']]
    heights = np.load(os.path.join(out, 'height.npy'))
    cell = rep['stack']['cellWorkPx'] / rep['reference']['scale']
    proj = PanoramaProjector(M, centres, D, heights, cell)

    pano = cv2.imread(os.path.join(out, a.image), cv2.IMREAD_COLOR)
    if pano is None:
        raise SystemExit(f'cannot read {os.path.join(out, a.image)}')
    if (pano.shape[1], pano.shape[0]) != (data['width'], data['height']):
        raise SystemExit(f'{a.image} is {pano.shape[1]}x{pano.shape[0]}, the stack canvas {data["width"]}x{data["height"]}')
    x0, y0, x1, y1 = aligned['board']['bbox']
    rect = (x0 - a.margin, y0 - a.margin, x1 + a.margin, y1 + a.margin)

    # texture
    t = time.time()
    tex = BoardTexture(rect, a.tex_size)
    img = tex.render(pano, M)
    cv2.imwrite(os.path.join(out, 'model_tex.jpg'), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    small = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(out, 'model_tex_4k.jpg'), small, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"texture {tex.W} x {tex.H} ({tex.s:.1f} px/mm) + {small.shape[1]} x {small.shape[0]}, {time.time() - t:.1f} s")

    # uv + photo eligibility
    t = time.time()
    V = model.vertices
    q = proj.project(V, parallax=not a.no_parallax)
    uv = tex.uvOfPano(q, M).astype(np.float32)
    vis = TopVisibility(model, rect)
    photo = (model.topSideVertices() & vis.visible(V)).astype(np.float32)
    shift = np.linalg.norm(q - proj.boardPx(V[:, :2].astype(np.float64)), axis=1) / pxPerMm * 1000 if not a.no_parallax else None
    print(f"uv + visibility in {time.time() - t:.1f} s; photo on {photo.mean() * 100:.0f} % of vertices"
          + (f"; parallax uv shift p50/p99/max {np.percentile(shift, 50):.0f}/{np.percentile(shift, 99):.0f}/"
             f"{shift.max():.0f} um" if shift is not None else ''))

    parts = model.partTable()

    def rgba(c):
        c = np.clip(c * 255 + 0.5, 0, 255).astype(np.uint8)
        return np.hstack([c, np.full((len(c), 1), 255, np.uint8)])
    grey = cv2.cvtColor(pano, cv2.COLOR_BGR2GRAY)
    check = RegistrationCheck(model, proj, grey)
    body = bodyColours(model, uv, img, check.topTriangles)      # photo colour of every part's top, for its walls
    colours, bodyRgba = rgba(model.colours), rgba(body[model.partOfVertex])
    F, U8, U16 = 5126, 5121, 5123
    GlbWriter().write(os.path.join(out, 'model.glb'), {
        'POSITION': (V, F, 'VEC3', False),
        'NORMAL': (model.normals, F, 'VEC3', False),
        'TEXCOORD_0': (uv, F, 'VEC2', False),
        '_STEPCOLOR': (colours, U8, 'VEC4', True),
        '_BODYCOLOR': (bodyRgba, U8, 'VEC4', True),
        '_PHOTO': (photo, F, 'SCALAR', False),
        '_PART': (model.partOfVertex, U16, 'SCALAR', False),
    }, model.triangles, 'model_tex.jpg', {'units': 'mm, z up (node rotates to y up, m)', 'rect': rect,
                                         'parts': [p['ref'] for p in parts]})

    # measured relief on the board mm grid; NaN outside the frames
    cover = cv2.resize(cv2.cvtColor(pano, cv2.COLOR_BGR2GRAY), (heights.shape[1], heights.shape[0]),
                       interpolation=cv2.INTER_AREA) > 3
    h, (rx0, ry1, rstep) = reliefGrid(proj, rect, a.relief_step, cover)
    hu = np.where(np.isnan(h), -32768, np.clip(np.round(h * 1000), -32000, 32000)).astype('<i2')
    hu.tofile(os.path.join(out, 'model_relief.bin'))

    # registration: CAD top outline vs photo edges, with and without the parallax correction
    t = time.time()
    del pano, img
    refs = [p['ref'] for p in parts if p['side'] == 'top' and p['fit'] == 'fitted' and p['bboxMax'][2] > 0.3]
    reg = check.run(refs)
    good = [r for r in reg if r['parallax']['score'] > 0.25]
    summary = {}
    for key in ('plain', 'parallax'):
        for lo, hi in ((0, 1.5), (1.5, 4), (4, 99)):
            s = [r[key]['shiftPx'] for r in good if lo <= r['heightMm'] < hi]
            if s:
                summary[f'{key} h {lo}-{hi} mm'] = {'parts': len(s), 'medianPx': float(np.median(s)),
                                                    'p90Px': float(np.percentile(s, 90)),
                                                    'medianUm': round(float(np.median(s)) / pxPerMm * 1000)}
    json.dump({'summary': summary, 'parts': reg}, open(os.path.join(out, 'model_registration.json'), 'w'), indent=1)
    print(f"registration ({len(good)}/{len(reg)} parts with a clear outline, {time.time() - t:.1f} s):")
    for k, v in summary.items():
        print(f"  {k}: {v['parts']} parts, shift median {v['medianPx']:.1f} px ({v['medianUm']} um), p90 {v['p90Px']:.1f} px")

    regByRef = {r['ref']: r['parallax'] for r in reg}
    for p in parts:
        if p['ref'] in regByRef:
            r = regByRef[p['ref']]
            p['registration'] = {'dxPx': r['dxPx'], 'dyPx': r['dyPx'], 'score': r['score']}
    meta = {'rect': rect, 'texture': {'full': 'model_tex.jpg', 'small': 'model_tex_4k.jpg', 'size': [tex.W, tex.H],
                                      'pxPerMm': tex.s},
            'relief': {'file': 'model_relief.bin', 'cols': int(h.shape[1]), 'rows': int(h.shape[0]), 'x0': rx0,
                       'y1': ry1, 'step': rstep},
            'parallax': not a.no_parallax, 'cameraDistanceMm': D, 'steep': a.steep, 'parts': parts,
            'board': os.path.basename(os.path.normpath(a.board))}
    with open(os.path.join(out, 'model_parts.json'), 'w', encoding='utf-8') as fp:
        json.dump(meta, fp, ensure_ascii=False, separators=(',', ':'))
    with open(os.path.join(out, 'model.html'), 'w', encoding='utf-8') as fp:
        fp.write(TEMPLATE.replace('__THREE__', THREE_VERSION))
    print(f"wrote model.glb, model_tex*.jpg, model_parts.json, model_relief.bin, model.html in {out}, "
          f"{time.time() - t0:.1f} s")


TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PCB CAD Textured</title>
<style>
  html, body { margin: 0; height: 100%; background: #111; color: #ddd; font: 13px system-ui, sans-serif; overflow: hidden; }
  #panel { position: absolute; top: 8px; left: 8px; background: rgba(20,20,20,.85); padding: 8px 10px;
           border-radius: 6px; z-index: 10; max-width: 330px; }
  #panel label { display: block; margin: 3px 0; cursor: pointer; }
  #panel input[type=range] { width: 120px; vertical-align: middle; }
  #panel button { background: #333; color: #ddd; border: 1px solid #555; border-radius: 3px; padding: 1px 7px; cursor: pointer; }
  #panel input[type=text] { width: 90px; background: #222; color: #ddd; border: 1px solid #555; }
  #info { position: absolute; top: 8px; right: 8px; background: rgba(20,20,20,.9); padding: 8px 10px;
          border-radius: 6px; max-width: 320px; display: none; white-space: pre-wrap; }
  #status { position: absolute; bottom: 8px; left: 8px; background: rgba(20,20,20,.85); padding: 4px 8px; border-radius: 4px; }
  small { color: #999; }
  a { color: #6cf; }
  hr { border: 0; border-top: 1px solid #444; margin: 6px 0; }
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
  <b>STEP model + photo</b> <a href="index.html" style="margin-left:8px">2D</a> <a href="3d.html" style="margin-left:6px">relief 3D</a><br>
  <small>focus.jpg projected from above onto the CAD mesh</small>
  <label><input type="radio" name="mode" value="2" checked> blend (photo on tops, STEP on walls)</label>
  <label><input type="radio" name="mode" value="0"> photo everywhere (raw projection)</label>
  <label><input type="radio" name="mode" value="1"> STEP colours</label>
  <label>steep below normal z <input type="range" id="steep" min="0" max="1" step="0.02"> <span id="steepV"></span></label>
  <label>walls: <select id="wall"><option value="3">body colour from photo</option><option value="0">STEP colour</option><option value="1">dim photo</option><option value="2">photo, blurred</option></select></label>
  <label>shading on photo <input type="range" id="shade" min="0" max="1" step="0.05" value="0.25"> <span id="shadeV"></span></label>
  <label>DNP / not in BOM: <select id="dnp"><option value="0">as is</option><option value="1" selected>tint</option><option value="2">hide</option></select></label>
  <label><input type="checkbox" id="hd"> 8k texture</label>
  <hr>
  <label><input type="checkbox" id="relief"> measured relief overlay</label>
  <label>opacity <input type="range" id="rOp" min="0.1" max="1" step="0.05" value="0.55"> <span id="rOpV"></span></label>
  <label><input type="checkbox" id="rWire"> wireframe</label>
  <hr>
  part <input type="text" id="find" placeholder="U1, J29..."> <button id="go">go</button>
  <div style="margin-top:5px"><button data-view="top">top</button> <button data-view="oblique">oblique</button>
    <button data-view="side">low side</button></div>
  <small>drag: rotate &middot; right-drag: pan &middot; wheel: zoom &middot; click: part &middot; double-click: orbit around point.
    Magenta: DNP / optional, blue: not in BOM, yellow: selected.</small>
</div>
<div id="info"></div>
<div id="status">loading&hellip;</div>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const status = document.getElementById('status');
const meta = await (await fetch('model_parts.json', { cache: 'no-cache' })).json();
const parts = meta.parts;

const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(innerWidth, innerHeight);
document.body.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a1a);
const camera = new THREE.PerspectiveCamera(35, innerWidth / innerHeight, 0.5, 2000);
camera.up.set(0, 0, 1);
const [rx0, ry0, rx1, ry1] = meta.rect;
const centre = new THREE.Vector3((rx0 + rx1) / 2, (ry0 + ry1) / 2, 0);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.copy(centre);
controls.enableDamping = true;
controls.screenSpacePanning = true;

// ---- GLB (own small reader: no material/texture fetch, custom attributes kept as they are)
async function loadGlb(url) {
  const buf = await (await fetch(url, { cache: 'no-cache' })).arrayBuffer();
  const dv = new DataView(buf);
  const jsonLen = dv.getUint32(12, true);
  const gltf = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 20, jsonLen)));
  const binOff = 20 + jsonLen + 8;
  const arrays = { 5126: Float32Array, 5125: Uint32Array, 5123: Uint16Array, 5121: Uint8Array };
  const size = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 };
  const acc = i => {
    const a = gltf.accessors[i], v = gltf.bufferViews[a.bufferView];
    const T = arrays[a.componentType];  // (accessors are tightly packed, 4-byte aligned)
    return { data: new T(buf, binOff + v.byteOffset, a.count * size[a.type]), n: size[a.type], norm: !!a.normalized };
  };
  const prim = gltf.meshes[0].primitives[0];
  const g = new THREE.BufferGeometry();
  const names = { POSITION: 'position', NORMAL: 'normal', TEXCOORD_0: 'uv', _STEPCOLOR: 'stepColor', _PHOTO: 'photo', _PART: 'part', _BODYCOLOR: 'bodyColor' };
  for (const [k, i] of Object.entries(prim.attributes)) {
    const a = acc(i);
    g.setAttribute(names[k] || k, new THREE.BufferAttribute(a.data, a.n, a.norm));
  }
  g.setIndex(new THREE.BufferAttribute(acc(prim.indices).data, 1));
  g.computeBoundingBox(); g.computeBoundingSphere();
  return g;
}

// ---- per-part flags: r = fit class (0 fitted / board, 1 DNP or optional, 2 not in BOM), g = selected
const partTex = new THREE.DataTexture(new Uint8Array(1024 * 4), 1024, 1, THREE.RGBAFormat);
function setPartFlags(selected) {
  const d = partTex.image.data;
  parts.forEach((p, i) => {
    d[i * 4] = (p.fit === 'nc' || p.fit === 'optional') ? 1 : (p.fit === 'none' ? 2 : 0);
    d[i * 4 + 1] = i === selected ? 255 : 0;
  });
  partTex.needsUpdate = true;
}
setPartFlags(-1);

const uniforms = {
  map: { value: null }, mode: { value: 2 }, steep: { value: meta.steep }, soft: { value: 0.08 },
  wall: { value: 3 }, shade: { value: 0.25 }, dnp: { value: 1 }, partTex: { value: partTex },
  lightDir: { value: new THREE.Vector3(0.35, -0.45, 0.82).normalize() },
};
const material = new THREE.ShaderMaterial({
  uniforms, glslVersion: THREE.GLSL3,
  vertexShader: `
    in vec4 stepColor; in vec4 bodyColor; in float photo; in float part;
    out vec2 vUv; out vec3 vN; out vec3 vStep; out vec3 vBody; out float vPhoto; flat out int vPart;
    void main() {
      vUv = uv; vN = normal; vStep = stepColor.rgb; vBody = bodyColor.rgb; vPhoto = photo; vPart = int(part + 0.5);
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }`,
  fragmentShader: `
    precision highp float;
    uniform sampler2D map; uniform sampler2D partTex; uniform int mode; uniform float steep; uniform float soft;
    uniform int wall; uniform float shade; uniform int dnp; uniform vec3 lightDir;
    in vec2 vUv; in vec3 vN; in vec3 vStep; in vec3 vBody; in float vPhoto; flat in int vPart;
    out vec4 outColor;
    vec3 toLinear(vec3 c) { return pow(c, vec3(2.2)); }
    void main() {
      vec4 flags = texelFetch(partTex, ivec2(vPart, 0), 0);
      int fit = int(flags.r * 255.0 + 0.5);
      if (dnp == 2 && fit > 0) discard;
      vec3 n = normalize(gl_FrontFacing ? vN : -vN);
      float lambert = 0.35 + 0.65 * max(dot(n, lightDir), 0.0);
      vec3 photoC = toLinear(texture(map, vUv).rgb);
      vec3 stepC = toLinear(vStep) * lambert;
      vec3 photoLit = photoC * mix(1.0, lambert * 1.15, shade);
      float w = 0.0;
      if (mode == 0) w = 1.0;
      else if (mode == 2) w = vPhoto * smoothstep(steep - soft, steep + soft, n.z);
      vec3 wallC = stepC;
      if (wall == 1) wallC = photoC * 0.45 * lambert + stepC * 0.3;
      if (wall == 2) wallC = toLinear(textureLod(map, vUv, 6.0).rgb) * lambert * 1.2;
      if (wall == 3) wallC = toLinear(vBody) * lambert * 1.1;
      if (mode == 1) wallC = stepC;
      vec3 c = mix(wallC, photoLit, w);
      if (dnp == 1 && fit == 1) c = mix(c, vec3(0.9, 0.1, 0.9), 0.55);
      if (dnp == 1 && fit == 2) c = mix(c, vec3(0.1, 0.6, 0.9), 0.25);
      if (flags.g > 0.0) c = mix(c, vec3(1.0, 0.85, 0.1), 0.45);
      outColor = vec4(pow(c, vec3(1.0 / 2.2)), 1.0);
    }`,
});

const geometry = await loadGlb('model.glb');
const mesh = new THREE.Mesh(geometry, material);
scene.add(mesh);

// ---- texture
const maxTex = renderer.capabilities.maxTextureSize;
const hdBox = document.getElementById('hd');
hdBox.disabled = maxTex < Math.max(...meta.texture.size);
hdBox.checked = !hdBox.disabled;
const loader = new THREE.TextureLoader();
let texName = '';
function loadTexture() {
  const name = hdBox.checked ? meta.texture.full : meta.texture.small;
  if (name === texName) return;
  texName = name;
  status.textContent = 'loading ' + name + '…';
  loader.load(name, t => {
    t.flipY = false; t.colorSpace = THREE.NoColorSpace;
    t.anisotropy = renderer.capabilities.getMaxAnisotropy();
    t.minFilter = THREE.LinearMipmapLinearFilter; t.generateMipmaps = true;
    if (uniforms.map.value) uniforms.map.value.dispose();
    uniforms.map.value = t; t.needsUpdate = true;
    const ppm = meta.texture.pxPerMm / (hdBox.checked ? 1 : 2);
    status.textContent = `${geometry.attributes.position.count.toLocaleString()} vertices, ${parts.length} parts, ` +
      `texture ${name} (${ppm.toFixed(0)} px/mm), uv parallax correction ${meta.parallax ? 'on' : 'off'}`;
    window.viewer.ready = true;
  });
}
hdBox.onchange = loadTexture;

// ---- measured relief (height.npy resampled to the board mm grid)
let relief = null;
const reliefMat = new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.55,
                                                side: THREE.DoubleSide, depthWrite: false });
async function buildRelief() {
  const R = meta.relief;
  const h = new Int16Array(await (await fetch(R.file, { cache: 'no-cache' })).arrayBuffer());
  const pos = new Float32Array(R.cols * R.rows * 3), col = new Float32Array(R.cols * R.rows * 3);
  const c = new THREE.Color();
  for (let j = 0; j < R.rows; j++) for (let i = 0; i < R.cols; i++) {
    const k = j * R.cols + i, v = h[k];
    const z = v === -32768 ? 0 : v / 1000;
    pos.set([R.x0 + i * R.step, R.y1 - j * R.step, z], k * 3);
    c.setHSL(0.66 - 0.66 * Math.min(Math.max(z / 4, 0), 1), 0.9, 0.5);
    col.set([c.r, c.g, c.b], k * 3);
  }
  const idx = [];
  for (let j = 0; j + 1 < R.rows; j++) for (let i = 0; i + 1 < R.cols; i++) {
    const a = j * R.cols + i, b = a + 1, d = a + R.cols, e = d + 1;
    if (h[a] === -32768 || h[b] === -32768 || h[d] === -32768 || h[e] === -32768) continue;
    idx.push(a, d, b, b, d, e);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('color', new THREE.BufferAttribute(col, 3));
  g.setIndex(idx);
  relief = new THREE.Mesh(g, reliefMat);
  relief.renderOrder = 1;
  scene.add(relief);
}
document.getElementById('relief').onchange = async e => {
  if (e.target.checked && !relief) await buildRelief();
  if (relief) relief.visible = e.target.checked;
};
document.getElementById('rWire').onchange = e => { reliefMat.wireframe = e.target.checked; };

// ---- controls
function bindRange(id, fn) {
  const el = document.getElementById(id), out = document.getElementById(id + 'V');
  const upd = () => { fn(+el.value); if (out) out.textContent = (+el.value).toFixed(2); };
  el.oninput = upd; upd();
}
document.getElementById('steep').value = meta.steep;
bindRange('steep', v => uniforms.steep.value = v);
bindRange('shade', v => uniforms.shade.value = v);
bindRange('rOp', v => reliefMat.opacity = v);
document.querySelectorAll('input[name=mode]').forEach(r => r.onchange = () => uniforms.mode.value = +r.value);
document.getElementById('wall').onchange = e => uniforms.wall.value = +e.target.value;
document.getElementById('dnp').onchange = e => uniforms.dnp.value = +e.target.value;
uniforms.dnp.value = +document.getElementById('dnp').value;

function setView(kind, target, dist) {
  const t = target || centre, d = dist || 150;
  controls.target.copy(t);
  if (kind === 'top') camera.position.set(t.x, t.y - 0.001, t.z + d);
  else if (kind === 'side') camera.position.set(t.x, t.y - d * 0.95, t.z + d * 0.25);
  else camera.position.set(t.x + d * 0.35, t.y - d * 0.6, t.z + d * 0.7);
  controls.update();
}
document.querySelectorAll('[data-view]').forEach(b => b.onclick = () => setView(b.dataset.view));
setView('oblique');

// ---- picking
const info = document.getElementById('info');
const ray = new THREE.Raycaster();
function showPart(i) {
  setPartFlags(i);
  if (i < 0) { info.style.display = 'none'; return; }
  const p = parts[i];
  const reg = p.registration ? `\nauto edge fit, rough: ${p.registration.dxPx}, ${p.registration.dyPx} px (score ${p.registration.score})` : '';
  info.textContent = `${p.ref}  [${p.side}, ${p.fit}]\n${p.comment}\n${p.package}\n${p.description}\n` +
    `STEP: ${p.step} (${p.leaves} bodies)\nz ${p.bboxMin[2]} … ${p.bboxMax[2]} mm${reg}`;
  info.style.display = 'block';
}
let down = null;
renderer.domElement.addEventListener('pointerdown', e => { down = [e.clientX, e.clientY]; });
function hit(e) {
  const m = new THREE.Vector2(e.clientX / innerWidth * 2 - 1, -e.clientY / innerHeight * 2 + 1);
  ray.setFromCamera(m, camera);
  for (const h of ray.intersectObject(mesh)) {
    const pi = geometry.attributes.part.getX(h.face.a);
    if (uniforms.dnp.value === 2 && parts[pi].fit !== 'fitted' && parts[pi].fit !== 'board') continue;
    return { part: pi, point: h.point };
  }
  return null;
}
renderer.domElement.addEventListener('pointerup', e => {
  if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4 || e.button !== 0) return;
  const h = hit(e);
  showPart(h ? h.part : -1);
});
renderer.domElement.addEventListener('dblclick', e => {
  const h = hit(e);
  if (h) { controls.target.copy(h.point); controls.update(); }
});
function focusPart(ref, kind) {
  const i = parts.findIndex(p => p.ref.toUpperCase() === ref.toUpperCase());
  if (i < 0) { status.textContent = ref + ' not in the model'; return; }
  showPart(i);
  const p = parts[i];
  const t = new THREE.Vector3((p.bboxMin[0] + p.bboxMax[0]) / 2, (p.bboxMin[1] + p.bboxMax[1]) / 2, Math.max(p.bboxMax[2], 0) / 2);
  const s = Math.max(p.bboxMax[0] - p.bboxMin[0], p.bboxMax[1] - p.bboxMin[1], 2);
  setView(kind || 'oblique', t, s * 3.5);
}
document.getElementById('go').onclick = () => focusPart(document.getElementById('find').value.trim());
document.getElementById('find').onkeydown = e => { if (e.key === 'Enter') focusPart(e.target.value.trim()); };

addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});
renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });
window.viewer = { THREE, camera, controls, scene, mesh, uniforms, parts, setView, showPart, focusPart, renderer, ready: false };
loadTexture();
</script>
</body>
</html>
'''


if __name__ == '__main__':
    main()
