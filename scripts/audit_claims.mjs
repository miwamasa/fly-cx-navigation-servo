// scripts/audit_claims.mjs — 誤差棒の棚卸し（迷路側）
//   実行:  node scripts/audit_claims.mjs
//   出力:  data/claims_audit_maze.json
//
// なぜこれが要るのか
// ------------------
// 本プロジェクトは「軌跡 6 本・単一シードの点推定」を陽性結果として論文に
// 書いてしまい、あとで再現しないことが分かった（PATH_INTEGRATION 6 節）。
// 同じ危険は迷路側にもある。閉ループの方位誤差も、Δ7 切除の効果も、
// PFL3 の釣り合い点も、いずれも 1 回走らせた値をそのまま文書に書いていた。
//
// そこでここでは、乱数シードと目標方位を振って分布を取り、
// 中央値とブートストラップ 95% 信頼区間を付け直す。
// 「壊れているか」ではなく「誤差棒を付けても同じことが言えるか」を確かめる。

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { GGUF } from '../web/gguf.js';
import { CXNetwork } from '../web/cxnet.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(HERE, '..');
const MODEL = path.join(ROOT, 'model', 'flybrain-cx.gguf');
const OUT = path.join(ROOT, 'data', 'claims_audit_maze.json');

const TAU = Math.PI * 2;
const wrap = (x) => ((x + Math.PI) % TAU + TAU) % TAU - Math.PI;
const deg = (r) => r * 180 / Math.PI;

function loadNet() {
  const b = fs.readFileSync(MODEL);
  const g = new GGUF(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
  return new CXNetwork(g);
}

// ---------------------------------------------------------------- 統計
function median(a) {
  const s = [...a].sort((x, y) => x - y);
  const n = s.length;
  return n % 2 ? s[(n - 1) / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2]);
}

/** 決定論的な線形合同法。seed を変えれば独立な再標本になる */
function rng(seed) {
  let s = seed >>> 0;
  return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
}

function bootstrapCI(values, stat = median, nBoot = 2000, seed = 12345) {
  const r = rng(seed), n = values.length, out = [];
  for (let b = 0; b < nBoot; b++) {
    const samp = new Array(n);
    for (let i = 0; i < n; i++) samp[i] = values[Math.floor(r() * n)];
    out.push(stat(samp));
  }
  out.sort((a, b) => a - b);
  return [out[Math.floor(0.025 * nBoot)], out[Math.floor(0.975 * nBoot)]];
}

function summarize(values) {
  const ci = bootstrapCI(values);
  return {
    median: median(values), ci95: ci, n: values.length,
    min: Math.min(...values), max: Math.max(...values),
  };
}

/** 絶対誤差と符号つき誤差の両方をまとめる。
 *  符号つきの中央値が 0 から離れていれば、残差は不感帯ではなく系統的な偏りである。 */
function summarizePair(rows) {
  const a = summarize(rows.map((r) => r.abs));
  const s = summarize(rows.map((r) => r.signed));
  return { abs: a, signed: s, median: a.median, ci95: a.ci95, n: a.n };
}

// ---------------------------------------------------------------- 閉ループ
/**
 * 目標方位へ向かわせて、最終的な残差を測る。
 * 初期状態のシードと目標方位を振って 1 試行ずつ回す。
 */
function closedLoopErrors(net, { seeds = 8, goals = 6, steps = 700 } = {}) {
  const errs = [];
  for (let s = 0; s < seeds; s++) {
    for (let k = 0; k < goals; k++) {
      // 目標は ±(30°〜170°) に散らす。0 付近だと初期状態のまま当たってしまう
      const mag = (30 + 140 * (k + 0.5) / goals) * Math.PI / 180;
      const goal = (k % 2 ? 1 : -1) * mag;
      net.reset(1 + s);
      let th = 0;
      for (let i = 0; i < 150; i++) {          // 助走（バンプを立てる）
        net.clearDrive(); net.setVisualScene(th, 1);
        net.injectGoal(goal, 0.3); net.step(0.02);
      }
      for (let i = 0; i < steps; i++) {
        net.clearDrive(); net.setVisualScene(th, 1);
        net.injectGoal(goal, 0.3); net.step(0.02);
        th = wrap(th + 2.6 * net.readSteering().turn * 0.02);
      }
      errs.push({ abs: Math.abs(deg(wrap(th - goal))), signed: deg(wrap(th - goal)) });
    }
  }
  return errs;
}

/** PFL3 の左右が釣り合う目標オフセット（＝直進する点）を二分法で探す */
function balancePoint(net, seed) {
  const turnAt = (off) => {
    net.reset(seed);
    for (let i = 0; i < 300; i++) {
      net.clearDrive(); net.setVisualScene(0, 1);
      net.injectGoal(off, 0.3); net.step(0.02);
    }
    return net.readSteering().turn;
  };
  let lo = -0.6, hi = 0.6;
  if (turnAt(lo) * turnAt(hi) > 0) return null;      // 符号が変わらないなら測れない
  for (let i = 0; i < 24; i++) {
    const mid = 0.5 * (lo + hi);
    if (turnAt(lo) * turnAt(mid) <= 0) hi = mid; else lo = mid;
  }
  return deg(0.5 * (lo + hi));
}

// ---------------------------------------------------------------- 本体
function main() {
  const net = loadNet();
  const all = net.aliveFromTypes(null);
  const result = { generated: new Date().toISOString(), claims: {} };
  const t0 = Date.now();

  // --- 1. ビット数と閉ループ方位誤差 ---
  console.log('1. ビット数 × 閉ループ方位誤差');
  const byBits = {};
  for (const bits of [8, 4, 3, 2]) {
    net.setAlive(all); net.applyWeightBudget(bits, 0);
    const e = closedLoopErrors(net);
    byBits[bits] = summarizePair(e);
    console.log(`   ${bits} bit  |誤差| ${byBits[bits].abs.median.toFixed(1)}° `
      + `[${byBits[bits].abs.ci95.map((v) => v.toFixed(1)).join(', ')}]`
      + `  符号つき ${byBits[bits].signed.median.toFixed(1)}°  n=${e.length}`
      + `  (${((Date.now() - t0) / 1000).toFixed(0)}s)`);
  }
  result.claims.bit_depth_heading_error_deg = byBits;

  // --- 2. Δ7 / ER の切除 ---
  console.log('2. Δ7 と ER の切除');
  const ablations = {
    '無傷': [],
    'Δ7 を切除': ['Delta7'],
    'ER を切除': ['ER'],
    'Δ7 と ER を切除': ['Delta7', 'ER'],
  };
  const byAbl = {};
  for (const [name, drop] of Object.entries(ablations)) {
    const mask = net.aliveFromTypes(null);
    for (let i = 0; i < net.N; i++) {
      const t = net.typeNames[net.typeId[i]] || '';
      if (drop.some((p) => t.startsWith(p))) mask[i] = 0;
    }
    net.setAlive(mask); net.applyWeightBudget(8, 0);
    const e = closedLoopErrors(net);
    byAbl[name] = summarizePair(e);
    console.log(`   ${name.padEnd(16)} |誤差| ${byAbl[name].abs.median.toFixed(1)}° `
      + `[${byAbl[name].abs.ci95.map((v) => v.toFixed(1)).join(', ')}]`
      + `  符号つき ${byAbl[name].signed.median.toFixed(1)}°`
      + `  (${((Date.now() - t0) / 1000).toFixed(0)}s)`);
  }
  result.claims.ablation_heading_error_deg = byAbl;

  // --- 3. PFL3 の釣り合い点 ---
  console.log('3. PFL3 の釣り合い点');
  net.setAlive(all); net.applyWeightBudget(8, 0);
  const bp = [];
  for (let s = 1; s <= 24; s++) {
    const v = balancePoint(net, s);
    if (v !== null) bp.push(v);
  }
  result.claims.pfl3_balance_point_deg = summarize(bp);
  console.log(`   中央値 ${median(bp).toFixed(2)}° `
    + `[${result.claims.pfl3_balance_point_deg.ci95.map((v) => v.toFixed(2)).join(', ')}]`
    + `  n=${bp.length}`);

  // --- 4. 最小回路の操舵が符号を間違えない率 ---
  console.log('4. 最小回路（EPG+FC2+PFL3, 4bit）の操舵');
  net.setAlive(net.aliveFromTypes(['EPG', 'PFL3', 'FC2']));
  net.applyWeightBudget(4, 0);
  const b = net.budget();
  let ok = 0, tot = 0;
  for (let s = 1; s <= 12; s++) {
    for (const goal of [0.9, -0.9, 1.6, -1.6]) {
      net.reset(s);
      for (let i = 0; i < 300; i++) {
        net.clearDrive(); net.setVisualScene(0, 1);
        net.injectGoal(goal, 0.3); net.step(0.02);
      }
      const t = net.readSteering().turn;
      tot++;
      if (Math.sign(t) === Math.sign(goal) && Math.abs(t) > 0.05) ok++;
    }
  }
  result.claims.minimal_circuit = {
    neurons: b.neurons, edges: b.edges, bits: b.bits, kbit: b.kbit,
    steering_sign_correct: ok, trials: tot, rate: ok / tot,
  };
  console.log(`   ${b.neurons} ニューロン / ${b.edges} シナプス / ${b.kbit.toFixed(2)} kbit`);
  console.log(`   操舵の符号が正しい: ${ok}/${tot}`);

  fs.writeFileSync(OUT, JSON.stringify(result, null, 2));
  console.log(`\n→ ${OUT}  (${((Date.now() - t0) / 1000).toFixed(0)}s)`);
}

main();
