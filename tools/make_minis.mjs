// Builds scenes3d/<task>/mini.glb: a light, low-poly copy of each 4D scene for the sandbox on the landing page.
// Walls and ceiling (walls.json) are taken out, each texture becomes its average colour (the page draws the
// tiles flat-shaded), meshes are merged and simplified, and the rigid and skeletal animation is kept.
//
// Setup (once, in this folder; node_modules is git-ignored):  npm i --no-save @gltf-transform/core@4 @gltf-transform/functions@4 @gltf-transform/extensions@4 meshoptimizer sharp
// Usage: node make_minis.mjs <scenes3d dir> [task prefix ...]   e.g.  cd tools && node make_minis.mjs ../scenes3d 05 11
import { NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS } from '@gltf-transform/extensions';
import { dequantize, flatten, join, weld, simplify, resample, prune, dedup, meshopt, palette } from '@gltf-transform/functions';
import { MeshoptDecoder, MeshoptEncoder, MeshoptSimplifier } from 'meshoptimizer';
import sharp from 'sharp';
import { readdirSync, readFileSync, existsSync, statSync } from 'node:fs';
import { join as pj } from 'node:path';

await Promise.all([MeshoptDecoder.ready, MeshoptEncoder.ready, MeshoptSimplifier.ready]);
const io = new NodeIO().registerExtensions(ALL_EXTENSIONS)
  .registerDependencies({ 'meshopt.decoder': MeshoptDecoder, 'meshopt.encoder': MeshoptEncoder });
const [root, ...only] = process.argv.slice(2);

// Merged room shells keep only their floor-level, upward-facing triangles (as the viewer's "hide walls" does).
function keepFloor(node) {
  const m = node.getWorldMatrix(), mesh = node.getMesh();
  if (!mesh) return;
  const tx = (p) => [m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12], m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13], m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14]];
  for (const prim of mesh.listPrimitives()) {
    const pos = prim.getAttribute('POSITION'), idx = prim.getIndices();
    const n = idx ? idx.getCount() : pos.getCount(), get = i => (idx ? idx.getScalar(i) : i);
    const keep = [], v = [];
    for (let i = 0; i < n; i += 3) {
      const t = [get(i), get(i + 1), get(i + 2)], [a, b, c] = t.map(k => tx(pos.getElement(k, v.slice())));
      const u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]], w = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
      const nx = u[1] * w[2] - u[2] * w[1], ny = u[2] * w[0] - u[0] * w[2], nz = u[0] * w[1] - u[1] * w[0];
      // glTF is Y-up: a floor faces +Y and sits below half a metre
      if (Math.abs(ny) > 0.7 * Math.hypot(nx, ny, nz) && (a[1] + b[1] + c[1]) / 3 < 0.5) keep.push(...t);
    }
    if (!keep.length) { prim.dispose(); continue; }
    prim.setIndices(idxDoc.createAccessor().setType('SCALAR').setArray(new Uint32Array(keep)));
  }
}

// gltf-transform's resample keeps a rotation key only once the path to it reaches half a turn, so a steady spin can
// end up with keys exactly half a turn apart, and slerp then turns either way round (19's recovery cam, 24's fly).
// The same check (keyframe-resample 0.1.0), with kept rotation keys at most a quarter turn apart.
function resampleQuarterTurn(input, output, interpolation, tolerance = 1e-4) {
  const n = output.length / input.length, get = (i, o) => { for (let k = 0; k < n; k++) o[k] = output[i * n + k]; return o; };
  const prev = new Array(n), value = new Array(n), next = new Array(n), tmp = new Array(n);
  const eq = (a, b, tol = 0) => a.every((x, k) => Math.abs(x - b[k]) <= tol);
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  const angle = (a, b) => { const d = dot(a, b); return Math.acos(Math.min(1, Math.max(-1, 2 * d * d - 1))); };
  const slerp = (a, b, t) => {
    let c = dot(a, b), s = c < 0 ? -1 : 1; c = Math.abs(c);
    let w0 = 1 - t, w1 = t;
    if (1 - c > 1e-6) { const om = Math.acos(c), so = Math.sin(om); w0 = Math.sin((1 - t) * om) / so; w1 = Math.sin(t * om) / so; }
    for (let k = 0; k < 4; k++) tmp[k] = w0 * a[k] + w1 * s * b[k];
    return tmp;
  };
  const last = input.length - 1;
  let w = 1;
  for (let i = 1; i < last; ++i) {
    const t0 = input[w - 1], t = input[i], t1 = input[i + 1], f = (t - t0) / (t1 - t0);
    let keep = false;
    if (t !== t1 && (i !== 1 || t !== input[0])) {
      get(w - 1, prev); get(i, value); get(i + 1, next);
      if (interpolation === 'slerp') keep = !eq(value, slerp(prev, next, f), tolerance) || angle(prev, value) + angle(value, next) >= Math.PI / 2;
      else if (interpolation === 'lerp') keep = !eq(value, prev.map((x, k) => x * (1 - f) + next[k] * f), tolerance);
      else keep = !eq(value, prev) || !eq(value, next);
    }
    if (keep) {
      if (i !== w) { input[w] = input[i]; for (let k = 0; k < n; k++) output[w * n + k] = output[i * n + k]; }
      w++;
    }
  }
  if (last > 0) { input[w] = input[last]; for (let k = 0; k < n; k++) output[w * n + k] = output[last * n + k]; w++; }
  return w;
}

let idxDoc;  // the document being built; keepFloor writes its new index accessors into it
const dirs = readdirSync(root).filter(d => existsSync(pj(root, d, 'scene.glb')) && (!only.length || only.some(p => d.startsWith(p))));
for (const d of dirs) {
  const src = pj(root, d, 'scene.glb'), dst = pj(root, d, 'mini.glb');
  const doc = await io.read(src); idxDoc = doc;
  const r = doc.getRoot();
  await doc.transform(dequantize());
  const wf = pj(root, d, 'walls.json');
  if (existsSync(wf)) {
    const w = JSON.parse(readFileSync(wf, 'utf8'));
    const byName = new Map(r.listNodes().map(n => [n.getName(), n]));
    w.hide.forEach(n => byName.get(n)?.dispose());
    w.cut.forEach(name => byName.get(name)?.traverse(keepFloor));
  }
  r.listCameras().forEach(c => c.dispose());
  // Morph targets (cooking food, pours) are most of the bytes and invisible at sandbox scale: keep the base shape.
  // (their samplers too: left behind, the per-frame weights of principal-shape morphs are megabytes)
  r.listAnimations().forEach(a => a.listChannels().forEach(c => {
    if (c.getTargetPath() !== 'weights') return;
    const s = c.getSampler();
    c.dispose();
    if (s && !a.listChannels().some(o => o.getSampler() === s)) s.dispose();
  }));
  r.listMeshes().forEach(m => { m.setWeights([]); m.listPrimitives().forEach(p => p.listTargets().forEach(t => t.dispose())); });
  // Textures become vertex colours; with UVs and normals gone, vertices weld and simplify properly.
  // Each vertex takes the colour under its UV. Small textures are palettes of flat blocks (the viewer's rule:
  // 256 px or less) and are read texel for texel; larger ones (characters' skin and clothes, wood, fabric) are
  // first averaged down to 128 px wide, weighting by alpha so the clear parts of hair cards and lashes add nothing.
  const toLinear = c => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
  const grids = new Map();
  const decode = async t => {
    if (grids.has(t)) return grids.get(t);
    let grid = null;
    try {
      const { data, info } = await sharp(Buffer.from(t.getImage())).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
      const W = info.width, H = info.height, palette = W <= 256 && H <= 256;
      const w = palette ? W : 128, h = palette ? H : Math.max(1, Math.round(128 * H / W));
      const sum = new Float64Array(w * h * 4), all = [0, 0, 0, 0];
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const k = (y * W + x) * 4, a = data[k + 3] / 255, c = (Math.floor(y * h / H) * w + Math.floor(x * w / W)) * 4;
        for (let j = 0; j < 3; j++) { const v = toLinear(data[k + j] / 255) * a; sum[c + j] += v; all[j] += v; }
        sum[c + 3] += a; all[3] += a;
      }
      const mean = all[3] > 0 ? all.slice(0, 3).map(v => v / all[3]) : [0.5, 0.5, 0.5];
      const rgb = new Float32Array(w * h * 3);
      for (let i = 0; i < w * h; i++) for (let j = 0; j < 3; j++) rgb[i * 3 + j] = sum[i * 4 + 3] > 1e-3 ? sum[i * 4 + j] / sum[i * 4 + 3] : mean[j];
      grid = { rgb, w, h, mean };
    } catch {}
    grids.set(t, grid);
    return grid;
  };
  for (const mesh of r.listMeshes()) for (const prim of mesh.listPrimitives()) {
    const m = prim.getMaterial(), t = m && m.getBaseColorTexture(), g = t && await decode(t);
    if (!g) continue;
    const n = prim.getAttribute('POSITION').getCount();
    const uv = prim.getAttribute('TEXCOORD_' + (m.getBaseColorTextureInfo().getTexCoord() || 0));
    const old = prim.getAttribute('COLOR_0'), col = new Float32Array(n * 4), e = [], c = [];
    for (let i = 0; i < n; i++) {
      let rgb = g.mean;
      if (uv) {
        uv.getElement(i, e);
        const x = Math.min(g.w - 1, Math.floor((e[0] - Math.floor(e[0])) * g.w));
        const y = Math.min(g.h - 1, Math.floor((e[1] - Math.floor(e[1])) * g.h));
        rgb = g.rgb.subarray((y * g.w + x) * 3, (y * g.w + x) * 3 + 3);
      }
      if (old) old.getElement(i, c); else c[0] = c[1] = c[2] = c[3] = 1;
      col.set([rgb[0] * c[0], rgb[1] * c[1], rgb[2] * c[2], c.length > 3 ? c[3] : 1], i * 4);
    }
    prim.setAttribute('COLOR_0', doc.createAccessor().setType('VEC4').setArray(col).setBuffer(r.listBuffers()[0]));
  }
  for (const m of r.listMaterials()) {
    m.setBaseColorTexture(null).setNormalTexture(null).setMetallicRoughnessTexture(null)
      .setOcclusionTexture(null).setEmissiveTexture(null);
  }
  r.listMeshes().forEach(m => m.listPrimitives().forEach(p => p.listSemantics().forEach(sem => {
    if (!/^(POSITION|JOINTS_0|WEIGHTS_0|COLOR_0)$/.test(sem)) p.setAttribute(sem, null);
  })));
  await doc.transform(
    dedup(),
    flatten(),
    join({ keepNamed: false }),
    weld(),
    simplify({ simplifier: MeshoptSimplifier, ratio: 0.05, error: 0.01, lockBorder: false }),
    palette({ min: 2 }),
    join({ keepNamed: false }),
    resample({ tolerance: 0.002, resample: resampleQuarterTurn }),
    prune(),
    meshopt({ encoder: MeshoptEncoder, level: 'medium' }),
  );
  await io.write(dst, doc);
  const tris = r.listMeshes().reduce((a, m) => a + m.listPrimitives().reduce((b, p) => b + (p.getIndices()?.getCount() ?? 0) / 3, 0), 0);
  console.log(d, (statSync(src).size / 1e6).toFixed(1) + 'MB ->', (statSync(dst).size / 1e6).toFixed(2) + 'MB',
    { meshes: r.listMeshes().length, nodes: r.listNodes().length, mats: r.listMaterials().length, tris: Math.round(tris) });
}
