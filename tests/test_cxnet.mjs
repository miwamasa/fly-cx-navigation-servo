// tests/test_cxnet.mjs — JS 側 (GGUF ローダ + 推論エンジン) のテスト
//   実行:  node tests/test_cxnet.mjs
//
// 「回路が文献どおりの計算をしているか」を回帰テストにしてある。
// パラメータを触ったときにここが落ちれば、生物学的な妥当性が壊れたということ。

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { GGUF, f16ToF32, dequantizeQ8_0 } from '../web/gguf.js';
import { CXNetwork, ROLE } from '../web/cxnet.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const MODEL = path.join(HERE, '..', 'model', 'flybrain-cx.gguf');

const TAU = Math.PI * 2;
const wrap = (x) => ((x + Math.PI) % TAU + TAU) % TAU - Math.PI;
const deg = (r) => r * 180 / Math.PI;

let pass = 0, fail = 0;
function check(name, cond, detail = '') {
  if (cond) { pass++; console.log(`  ok   ${name}${detail ? '  (' + detail + ')' : ''}`); }
  else { fail++; console.log(`  FAIL ${name}${detail ? '  (' + detail + ')' : ''}`); }
}
function group(t) { console.log(`\n${t}`); }

function loadNet() {
  const b = fs.readFileSync(MODEL);
  const g = new GGUF(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
  return { g, net: new CXNetwork(g) };
}

/** 視覚手がかりを theta に固定して n ステップ回す */
function settle(net, theta, steps = 250, goal = null, clarity = 1) {
  for (let k = 0; k < steps; k++) {
    net.clearDrive();
    net.setVisualScene(theta, clarity);
    if (goal !== null) net.injectGoal(goal, 0.3);
    net.step(0.02);
  }
}

// ---------------------------------------------------------------- 1. f16
group('1. 数値プリミティブ');
{
  const cases = [[0x3C00, 1], [0xBC00, -1], [0x0000, 0], [0x4000, 2], [0x3555, 0.333251953125]];
  let ok = true;
  for (const [h, v] of cases) if (Math.abs(f16ToF32(h) - v) > 1e-6) ok = false;
  check('f16ToF32 が既知の値を再現する', ok);

  // Q8_0 の逆量子化: ブロックを手で組んで確認
  const buf = new Uint8Array(34);
  new DataView(buf.buffer).setUint16(0, 0x3C00, true);   // scale = 1.0
  for (let i = 0; i < 32; i++) buf[2 + i] = (i - 16) & 0xff;
  const d = dequantizeQ8_0(buf, 0, 32);
  check('Q8_0 逆量子化が int8 × scale になる',
    d[0] === -16 && d[31] === 15, `d[0]=${d[0]} d[31]=${d[31]}`);
}

// ---------------------------------------------------------------- 2. GGUF
group('2. GGUF の読み込み');
const { g, net } = loadNet();
{
  check('アーキテクチャ識別子', g.kv['general.architecture'] === 'connectome-cx');
  check('ライセンス表記がある', g.kv['general.license'] === 'CC-BY-4.0');
  check('出典 URL がある', String(g.kv['general.source.url']).includes('male-cns'));
  check('ニューロン数が 2308', g.kv['cx.neuron_count'] === 2308, String(g.kv['cx.neuron_count']));
  check('量子化形式が Q8_0', g.kv['cx.quantization'] === 'Q8_0');

  const ip = net.indptr;
  check('CSR indptr が単調非減少',
    Array.from(ip).every((v, i) => i === 0 || v >= ip[i - 1]));
  check('CSR indptr の末尾が辺数と一致',
    ip[net.N] === g.kv['cx.edge_count'], `${ip[net.N]} vs ${g.kv['cx.edge_count']}`);
  check('全ての pre インデックスが範囲内',
    Array.from(net.indices.slice(0, ip[net.N])).every((v) => v >= 0 && v < net.N));
  check('量子化重みが [-1,1] に収まる',
    Array.from(net.weights.slice(0, ip[net.N])).every((v) => Math.abs(v) <= 1.0001));
  check('Q8_0 のブロック長 32 にパディングされている',
    net.weights.length % 32 === 0);
}

// ---------------------------------------------------------------- 3. 構成
group('3. 回路の構成');
{
  check('EPG が 50 個', net.byRole.get(ROLE.EPG).length === 50);
  check('PFL3 が左右 12 個ずつ',
    net.pfl3L.length === 12 && net.pfl3R.length === 12);
  check('PEN が左右に分かれている', net.penL.length > 0 && net.penR.length > 0);
  check('目標集団 (FC2/hDeltaB) がある', net.goalCells.length > 50, String(net.goalCells.length));
  check('PFL3 の左右オフセットが ±60〜85° で対称',
    Math.abs(deg(net.pfl3OffsetLeft) + 73) < 12 && Math.abs(deg(net.pfl3OffsetRight) - 73) < 12,
    `L=${deg(net.pfl3OffsetLeft).toFixed(1)}° R=${deg(net.pfl3OffsetRight).toFixed(1)}°`);
  check('直進となる目標オフセット (null) がほぼ 0°',
    Math.abs(deg(net.goalNullOffset)) < 15, `${deg(net.goalNullOffset).toFixed(1)}°`);
  check('入力層 (ER/ExR/LNO/SpsP) がクランプされている',
    net.clamped.reduce((a, b) => a + b, 0) > 300);
}

// ---------------------------------------------------------------- 4. バンプ
group('4. EPG の方位バンプ');
{
  let maxErr = 0, minStr = 1;
  for (const th of [-2.0, -0.8, 0, 0.8, 2.0, 3.0]) {
    net.reset(3); settle(net, th);
    const h = net.readHeading();
    maxErr = Math.max(maxErr, Math.abs(wrap(h.theta - th)));
    minStr = Math.min(minStr, h.strength);
  }
  check('バンプが手がかり方位に立つ (誤差 < 30°)', maxErr < 0.52, `最大 ${deg(maxErr).toFixed(1)}°`);
  check('バンプが局在している (鋭さ > 0.4)', minStr > 0.4, `最小 ${minStr.toFixed(2)}`);

  net.reset(3); settle(net, 0);
  const prof = net.ringProfile(16);
  const peak = prof.indexOf(Math.max(...prof));
  check('リング活動のピークが 0° 付近のビン',
    peak === 0 || peak === 15 || peak === 1, `bin=${peak}`);

  // 発火率が生理的な範囲か（暴走していないか）
  const s = net.stats();
  check('回路全体が飽和していない', s.meanRate < 0.35, `平均 ${s.meanRate.toFixed(3)}`);
}

// ---------------------------------------------------------------- 5. 操舵
group('5. PFL3 の操舵計算');
{
  const offs = [-90, -60, -30, 30, 60];
  const turns = offs.map((d) => {
    net.reset(3); settle(net, 0, 300, d * Math.PI / 180);
    return net.readSteering().turn;
  });
  console.log('     goal-head(deg):', offs.join('  '));
  console.log('     turn          :', turns.map((x) => x.toFixed(2)).join(' '));

  check('目標が反時計回り側なら turn > 0',
    turns[3] > 0.05 && turns[4] > 0.05, `+30:${turns[3].toFixed(2)} +60:${turns[4].toFixed(2)}`);
  check('目標が時計回り側なら turn < 0',
    turns[0] < -0.05 && turns[1] < -0.05, `-90:${turns[0].toFixed(2)} -60:${turns[1].toFixed(2)}`);
  check('反対称性 (符号が左右で反転)',
    Math.sign(turns[1]) !== Math.sign(turns[4]));

  net.reset(3); settle(net, 0, 300, 0);
  const straight = net.readSteering().turn;
  check('目標＝方位のとき ほぼ直進', Math.abs(straight) < 0.2, straight.toFixed(3));
}

// ---------------------------------------------------------------- 6. 閉ループ
group('6. 閉ループ（ハエの脳が目標方位へ向く）');
{
  for (const goal of [1.2, -1.2, 2.5]) {
    net.reset(3);
    let th = 0;
    settle(net, th, 150, goal);
    for (let k = 0; k < 700; k++) {
      net.clearDrive();
      net.setVisualScene(th, 1);
      net.injectGoal(goal, 0.3);
      net.step(0.02);
      th = wrap(th + 2.6 * net.readSteering().turn * 0.02);
    }
    const err = Math.abs(wrap(th - goal));
    check(`目標 ${deg(goal).toFixed(0)}° に収束する`, err < 0.6, `残差 ${deg(err).toFixed(1)}°`);
  }
}

// ---------------------------------------------------------------- 7. 最小の脳
group('7. 細胞の除去と重みの情報量削減');
{
  const full = net.budget();
  check('無傷の情報量が約 1,496 kbit',
    Math.abs(full.kbit - 1495.8) < 1, full.kbit.toFixed(1));

  // ビット数を落とすと、行内最大値に対して小さすぎる重みは 0 に丸められる。
  // つまり量子化そのものが弱いシナプスの剪定として働く（保存も不要になる）。
  net.applyWeightBudget(4, 0);
  const q4 = net.budget();
  check('4bit 量子化が弱いシナプスを 0 に丸めて実質剪定する',
    q4.edges < full.edges * 0.7 && q4.edges > full.edges * 0.3,
    `${q4.edges} / ${full.edges} (${(100 * q4.edges / full.edges).toFixed(0)}%)`);
  check('4bit の情報量が 8bit の 1/3 以下になる',
    q4.kbit < full.kbit / 3, `${q4.kbit.toFixed(1)} vs ${full.kbit.toFixed(1)} kbit`);

  // 足切りはシナプスを減らす
  net.applyWeightBudget(8, 0.1);
  check('足切り 0.1 でシナプスが半分以下になる',
    net.budget().edges < full.edges * 0.5, String(net.budget().edges));

  // 元に戻せる
  net.applyWeightBudget(8, 0);
  check('元の重みに戻せる（再量子化は非破壊）',
    net.budget().edges === full.edges && Math.abs(net.budget().kbit - full.kbit) < 0.01);

  // 既知の最小回路が課題を解けること（README の記録と一致させる）
  net.setAlive(net.aliveFromTypes(['EPG', 'PFL3', 'FC2']));
  net.applyWeightBudget(4, 0);
  const b = net.budget();
  check('最小回路が 166 ニューロン / 932 シナプス / 3.7 kbit',
    b.neurons === 166 && b.edges === 932 && Math.abs(b.kbit - 3.728) < 0.01,
    `${b.neurons} / ${b.edges} / ${b.kbit.toFixed(1)} kbit`);

  net.reset(3); settle(net, 0, 300, 0.9);
  const t1 = net.readSteering().turn;
  net.reset(3); settle(net, 0, 300, -0.9);
  const t2 = net.readSteering().turn;
  check('最小回路でも操舵の符号が正しい (元の 1/401 の情報量)',
    t1 > 0.05 && t2 < -0.05, `+51°:${t1.toFixed(2)} -51°:${t2.toFixed(2)}`);

  // FC2 を落とすと操舵が死ぬ = 目標入力が必須であること
  net.setAlive(net.aliveFromTypes(['EPG', 'PFL3']));
  net.applyWeightBudget(8, 0);
  net.reset(3); settle(net, 0, 300, 0.9);
  const s3 = net.readSteering();
  check('FC2 を切ると PFL3 が発火しなくなる（目標入力が必須）',
    s3.left + s3.right < 0.05, (s3.left + s3.right).toFixed(3));

  // 2bit まで削ると壊れる（崖があること）
  net.setAlive(net.aliveFromTypes(null));
  net.applyWeightBudget(2, 0);
  net.reset(3); settle(net, 0, 300, 0.9);
  const t4 = net.readSteering().turn;
  check('2bit まで削ると操舵が壊れる', !(t4 > 0.05), t4.toFixed(3));

  net.setAlive(net.aliveFromTypes(null));
  net.applyWeightBudget(8, 0);
}

// ---------------------------------------------------------------- 8. 性能
group('8. 性能');
{
  net.reset(3);
  const t0 = performance.now();
  for (let k = 0; k < 200; k++) { net.clearDrive(); net.setVisualScene(0, 1); net.step(0.02); }
  const ms = (performance.now() - t0) / 200;
  check('1 ステップ 10ms 未満（60fps で回せる）', ms < 10, `${ms.toFixed(2)} ms/step`);
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
