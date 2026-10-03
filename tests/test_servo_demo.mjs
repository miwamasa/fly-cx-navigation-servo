// tests/test_servo_demo.mjs — checks for the "Fly Servo Lab" browser demo (web/servo.html).
//
//   node tests/test_servo_demo.mjs          (about 40 seconds)
//
// The demo runs paper B section 7's added circuit in the browser with web/servo_augmented.js,
// a JavaScript port of scripts/servo_augmented.py on top of the same engine (web/cxnet.js)
// and the same model file. This test checks
//   * the page is English and its assets exist;
//   * web/servo_demo_data.json carries the configuration evaluated in paper B
//     (data/servo_augmented.json), and the alignment weights M are non-negative and sit
//     only on PFN -> hDeltaB pairs that exist in the connectome;
//   * with every added mechanism off, the fly is exactly the plain circuit;
//   * the paper's reference task (12 sideslip trials, seed 20260919) run in JavaScript gives
//     the same medians as Python (42.6 / 17.7 / 14.8 deg) within 1.5 deg, and every trial's
//     median within 0.5 deg. Rounding differences between the two engines are the only
//     expected source of difference (observed: below 0.01 deg);
//   * switching one mechanism off reproduces the ablations of paper B section 7;
//   * web/servo-standalone.html embeds exactly the current page, engine, model and data.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import os from 'node:os';
import { GGUF } from '../web/gguf.js';
import { ServoFly, runTrial, median } from '../web/servo_augmented.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(HERE, '..');
const WEB = path.join(ROOT, 'web');
const PAGE = path.join(WEB, 'servo.html');
const STANDALONE = path.join(WEB, 'servo-standalone.html');
const MODEL = path.join(ROOT, 'model', 'flybrain-cx.gguf');
const TOL_DEG = 1.5;          // pooled median, as stated in the README
const TOL_TRIAL_DEG = 0.5;    // per trial (observed: below 0.01 deg)

let pass = 0, fail = 0;
function check(name, cond, detail = '') {
  if (cond) { pass++; console.log(`  ok   ${name}`); }
  else { fail++; console.log(`  FAIL ${name} ${detail}`); }
}

const html = fs.readFileSync(PAGE, 'utf8');
const data = JSON.parse(fs.readFileSync(path.join(WEB, 'servo_demo_data.json'), 'utf8'));
const saved = JSON.parse(fs.readFileSync(path.join(ROOT, 'data', 'servo_augmented.json'), 'utf8'));
const b = fs.readFileSync(MODEL);
const gguf = new GGUF(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));

console.log('page');
check('page is English (no CJK characters)', !/[぀-ヿ㐀-䶿一-鿿＀-￯]/.test(html));
check('page loads the shared engine and the port',
  html.includes("from './gguf.js'") && html.includes("from './servo_augmented.js'"));
check('page loads the model from ../model/', html.includes("GGUF.fromURL('../model/flybrain-cx.gguf')"));
check('page loads web/servo_demo_data.json', html.includes("await (await fetch('./servo_demo_data.json')).json()"));
for (const f of ['gguf.js', 'cxnet.js', 'servo_augmented.js', 'servo_demo_data.json', 'servo-standalone.html'])
  check(`web/${f} exists`, fs.existsSync(path.join(WEB, f)));
check('page does not use numbers from outside this work', !/GPT-6|Astra|15\.14|0\.921|160\.49/.test(html));

console.log('exported configuration = data/servo_augmented.json');
for (const v of ['N', 'C']) {
  const d = data.variants[v], s = saved.variants[v];
  check(`${v}: m ${d.m}, nu_P ${d.nu_P}, nu_B ${d.nu_B}, lambda ${d.lam}`,
    d.m === s.config.m && d.nu_P === s.config.nu_P && d.nu_B === s.config.nu_B && d.lam === s.config.lam);
  check(`${v}: c_T ${(d.c_T * 180 / Math.PI).toFixed(2)} deg`,
    Math.abs(d.c_T * 180 / Math.PI - s.config.c_T_deg) < 1e-9);
  check(`${v}: ${d.M.length} non-zero alignment weights (paper: ${s.M_nonzero})`,
    d.M.length === s.M_nonzero && d.M_nonzero === s.M_nonzero);
}
for (const k of ['plain', 'N', 'C']) {
  const py = k === 'plain' ? saved.plain.sideslip
    : saved.variants[k].evaluation.closed_loop.sideslip;
  check(`reference median for ${k} is the paper's ${py.median_deg.toFixed(1)} deg`,
    data.reference.python_median_deg[k] === py.median_deg);
}
check('12 reference tracks of 900 steps',
  data.reference.tracks.length === 12 && data.reference.tracks.every((t) => t.length === 900));

console.log('alignment weights M');
{
  const fly = new ServoFly(gguf, data, { variant: 'N' });
  const { net } = fly;
  const has = (post, pre) => {
    for (let e = net.indptr[post]; e < net.indptr[post + 1]; e++)
      if (net.indices[e] === pre && net.weights[e] !== 0) return true;
    return false;
  };
  for (const v of ['N', 'C']) {
    const M = data.variants[v].M;
    check(`${v}: all weights > 0`, M.every(([, , w]) => w > 0));
    const bad = M.filter(([a, j]) => !has(fly.hdb[a], fly.pfn[j]));
    check(`${v}: every weight sits on an existing PFN -> hDeltaB connection`, bad.length === 0,
      `${bad.length} not`);
  }
}

console.log('mechanisms off = plain circuit');
{
  const plain = new ServoFly(gguf, data);
  const off = new ServoFly(gguf, data, { variant: 'C', mechs: [] });
  const G = 1.0, track = data.reference.tracks[0];
  let maxd = 0;
  for (let i = 0; i < 300; i++) {
    plain.step(G, track[i], 1); off.step(G, track[i], 1);
    for (let k = 0; k < plain.net.N; k++) maxd = Math.max(maxd, Math.abs(plain.net.r[k] - off.net.r[k]));
  }
  check('variant C with no mechanisms reproduces the plain circuit exactly',
    maxd === 0 && plain.H === off.H, `max |dr| ${maxd}`);
  const on = new ServoFly(gguf, data, { variant: 'C' });
  check('switching the model on does not change the plain network (separate gain tables)',
    on.net.rowGain !== plain.net.rowGain);
}

console.log('reference task (paper B section 6/7): 12 trials x 18 s, error after 3 s');
for (const [k, opt] of [['plain', {}], ['N', { variant: 'N' }], ['C', { variant: 'C' }]]) {
  const fly = new ServoFly(gguf, data, opt);
  const all = [], per = [];
  const t0 = Date.now();
  for (let t = 0; t < data.reference.tracks.length; t++) {
    const e = runTrial(fly, data, t);
    all.push(...e); per.push(median(e));
  }
  const js = median(all), py = data.reference.python_median_deg[k];
  const pyPer = data.reference.python_per_trial_median_deg[k];
  const dPer = Math.max(...per.map((x, i) => Math.abs(x - pyPer[i])));
  check(`${k}: JS median ${js.toFixed(2)} deg vs Python ${py.toFixed(2)} deg (tolerance ${TOL_DEG})`,
    Math.abs(js - py) <= TOL_DEG);
  check(`${k}: every trial's median within ${TOL_TRIAL_DEG} deg of Python (largest difference ${dPer.toFixed(3)} deg, ${((Date.now() - t0) / 1e3).toFixed(1)} s)`,
    dPer <= TOL_TRIAL_DEG);
}

console.log('the page\'s toggles = the ablations of paper B section 7 (12 reference trials each)');
for (const v of ['N', 'C']) {
  for (const k of [1, 2, 3]) {
    const name = ['without (1) multiplication', 'without (2) alignment', 'without (3) comparator'][k - 1];
    const py = saved.variants[v].ablations[name].sideslip.median_deg;
    const fly = new ServoFly(gguf, data, { variant: v, mechs: [1, 2, 3].filter((x) => x !== k) });
    const all = [];
    for (let t = 0; t < data.reference.tracks.length; t++) all.push(...runTrial(fly, data, t));
    const js = median(all);
    check(`${v} ${name}: JS ${js.toFixed(2)} deg vs Python ${py.toFixed(2)} deg`, Math.abs(js - py) <= TOL_DEG);
  }
}

console.log('standalone file');
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'servo-'));
const rebuilt = path.join(tmp, 'servo-standalone.html');
execFileSync('python3', [path.join(ROOT, 'scripts', 'bundle.py'), '--page', 'servo.html',
  '--module', 'servo_augmented.js', '--data', 'servo_demo_data.json', '-o', rebuilt], { stdio: 'ignore' });
const same = fs.readFileSync(rebuilt).equals(fs.readFileSync(STANDALONE));
check('web/servo-standalone.html is up to date (rebuild with scripts/bundle.py, see README)', same);
const sa = fs.readFileSync(STANDALONE, 'utf8');
check('standalone embeds the model', sa.includes(b.toString('base64').slice(0, 4000)));
check('standalone has no module imports and does not fetch the data', !/^\s*import\s/m.test(sa) && !sa.includes("fetch('./servo_demo_data.json')"));
fs.rmSync(tmp, { recursive: true, force: true });

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
