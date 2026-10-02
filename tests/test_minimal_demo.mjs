// tests/test_minimal_demo.mjs — checks for the "Minimal Brain" browser demo (web/minimal.html).
//
//   node tests/test_minimal_demo.mjs
//
// The demo shows a "current record" and the size of the intact brain. Both are computed here
// with the same engine (web/cxnet.js) and the same operations the page performs (setAlive by
// cell type, applyWeightBudget), so the numbers on the page cannot drift from the model.
// It also checks that the page is English, that its assets exist, and that the standalone
// file embeds exactly the current model and engine (rebuild it with scripts/bundle.py).

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import os from 'node:os';
import { GGUF } from '../web/gguf.js';
import { CXNetwork } from '../web/cxnet.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(HERE, '..');
const PAGE = path.join(ROOT, 'web', 'minimal.html');
const STANDALONE = path.join(ROOT, 'web', 'minimal-standalone.html');
const MODEL = path.join(ROOT, 'model', 'flybrain-cx.gguf');

let pass = 0, fail = 0;
function check(name, cond, detail = '') {
  if (cond) { pass++; console.log(`  ok   ${name}`); }
  else { fail++; console.log(`  FAIL ${name} ${detail}`); }
}

const html = fs.readFileSync(PAGE, 'utf8');
const b = fs.readFileSync(MODEL);
const net = new CXNetwork(new GGUF(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)));

console.log('page');
check('page is English (no CJK characters)', !/[぀-ヿ㐀-䶿一-鿿＀-￯]/.test(html));
check('page loads the shared engine', html.includes("from './gguf.js'") && html.includes("from './cxnet.js'"));
check('page loads the model from ../model/', html.includes("GGUF.fromURL('../model/flybrain-cx.gguf')"));

console.log('numbers shown on the page');
net.setAlive(net.aliveFromTypes(null)); net.applyWeightBudget(8, 0);
const full = net.budget();
const fullKbit = Number(/const FULL_KBIT = ([\d.]+);/.exec(html)[1]);
check(`intact size ${full.kbit.toFixed(3)} kbit equals FULL_KBIT ${fullKbit}`, Math.abs(full.kbit - fullKbit) < 1e-3);
check('footer says 1,496 kbit', html.includes('The intact brain is 1,496 kbit') && Math.round(full.kbit) === 1496);

net.setAlive(net.aliveFromTypes(['EPG', 'PFL3', 'FC2'])); net.applyWeightBudget(4, 0);
const m = net.budget();
check(`record circuit is ${m.neurons} neurons / ${m.edges} synapses / ${m.kbit.toFixed(1)} kbit`,
  m.neurons === 166 && m.edges === 932 && m.kbit.toFixed(1) === '3.7');
check('page states paper A\'s circuit', html.includes('<b>166 neurons / 932 synapses / 4 bit = 3.7 kbit</b>'));
check(`ratio 1/${(full.kbit / m.kbit).toFixed(0)} matches the page`,
  html.includes(`1/${(full.kbit / m.kbit).toFixed(0)} of the intact brain`));

console.log('steering of the record circuit (open loop, as in scripts/audit_claims.mjs)');
let ok = 0, tot = 0;
for (let s = 1; s <= 6; s++) {
  for (const goal of [0.9, -0.9, 1.6, -1.6]) {
    net.reset(s);
    for (let i = 0; i < 300; i++) { net.clearDrive(); net.setVisualScene(0, 1); net.injectGoal(goal, 0.3); net.step(0.02); }
    const t = net.readSteering().turn;
    tot++; if (Math.sign(t) === Math.sign(goal) && Math.abs(t) > 0.05) ok++;
  }
}
check(`steering sign correct ${ok}/${tot}`, ok === tot);

console.log('standalone file');
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'minimal-'));
const rebuilt = path.join(tmp, 'minimal-standalone.html');
execFileSync('python3', [path.join(ROOT, 'scripts', 'bundle.py'), '--page', 'minimal.html', '-o', rebuilt],
  { stdio: 'ignore' });
const same = fs.readFileSync(rebuilt).equals(fs.readFileSync(STANDALONE));
check('web/minimal-standalone.html is up to date (rebuild with scripts/bundle.py)', same);
check('standalone embeds the model', fs.readFileSync(STANDALONE, 'utf8').includes(b.toString('base64').slice(0, 4000)));
fs.rmSync(tmp, { recursive: true, force: true });

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
