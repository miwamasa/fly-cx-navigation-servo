// web/servo_augmented.js — paper B section 7 in the browser.
//
// A JavaScript port of scripts/servo_augmented.py (classes Augmentation and the closed loop)
// and of BodyVelocityInput in scripts/coord_transform.py, on top of the same engine
// (web/cxnet.js) and the same model file. The calibrated configuration of both variants,
// the alignment weights M and the reference sideslip tracks come from
// web/servo_demo_data.json (written by scripts/export_servo_demo.py).
//
// tests/test_servo_demo.mjs runs the paper's reference task with this code and checks the
// medians against the Python results.

import { CXNetwork } from './cxnet.js';

export const GROUPS = ['PFNd-L', 'PFNd-R', 'PFNv-L', 'PFNv-R'];
const TAU = 2 * Math.PI;
export const wrap = (a) => ((a + Math.PI) % TAU + TAU) % TAU - Math.PI;

/** Population-vector angle of rates r over cells with phases ph (null if silent). */
function popAngle(r, idx, ph) {
  let x = 0, y = 0;
  for (let k = 0; k < idx.length; k++) {
    const a = r[idx[k]];
    x += a * Math.cos(ph[k]);
    y += a * Math.sin(ph[k]);
  }
  const mag = Math.hypot(x, y);
  return { angle: mag > 1e-12 ? Math.atan2(y, x) : null, mag };
}

/** Dense post x pre block of the connectome's weights, negative entries clipped to 0. */
function block(net, post, pre) {
  const col = new Map(pre.map((j, k) => [j, k]));
  const out = post.map(() => new Float64Array(pre.length));
  post.forEach((i, a) => {
    for (let e = net.indptr[i]; e < net.indptr[i + 1]; e++) {
      const k = col.get(net.indices[e]);
      if (k !== undefined) out[a][k] = Math.max(0, net.weights[e]);
    }
  });
  return out;
}

/**
 * One simulated fly: the connectome network, its self-motion input and, optionally,
 * the added circuit of paper B section 7.
 *
 *   variant  null (plain circuit), 'N' (decoded angles) or 'C' (circuit only)
 *   mechs    which added mechanisms are on: subset of [1, 2, 3]
 */
export class ServoFly {
  constructor(gguf, data, { variant = null, mechs = [1, 2, 3] } = {}) {
    const net = new CXNetwork(gguf);
    // the model file's tensors are cached and shared between networks: copy before scaling
    net.rowGain = Float32Array.from(net.rowGain);
    net.goalCells = net._byTypePrefix(['FC2']);          // goal into FC2 only (paper B 5.2)
    this.net = net;
    this.data = data;
    this.variant = variant;
    this.mechs = new Set(variant ? mechs : []);
    const c = data.cells;
    this.groups = GROUPS.map((g) => c.groups[g]);
    this.pfn = this.groups.flat();
    this.pref = GROUPS.map((g) => (data.pref_deg[g] * Math.PI) / 180);
    this.hdb = c.hdb;
    this.psi = c.psi;
    this.fc2 = c.fc2;
    this.fc2Phase = c.fc2.map((i) => net.phase[i]);
    this.pfl3 = [...c.pfl3_left, ...c.pfl3_right];
    this.side = [...c.pfl3_left.map(() => -1), ...c.pfl3_right.map(() => 1)];
    this.gamma = this.pfl3.map((i) => net.phase[i]);
    this.delta = this.side.map((s) => (s < 0 ? Math.PI / 2 : -Math.PI / 2));
    this.routes = data.bvi_routes;

    if (variant) {
      const v = data.variants[variant];
      this.cfg = v;
      if (this.mechs.has(1) && v.nu_P !== 1) for (const i of this.pfn) net.rowGain[i] *= v.nu_P;
      if (this.mechs.has(2) && v.nu_B !== 1) for (const i of this.hdb) net.rowGain[i] *= v.nu_B;
      this.M = v.M;                                            // [hdb index, pfn index, weight]
      if (variant === 'C') {
        this.Wpe = block(net, this.pfn, net.epg);
        this.Wpf = block(net, this.pfl3, this.fc2);
        const o = this.side.map((s) => (s < 0 ? c.pfl3_offset.left : c.pfl3_offset.right));
        this.PT = this.pfl3.map((_, a) => this.psi.map((p) =>
          Math.cos(p - v.c_T - this.gamma[a] - o[a] - this.delta[a])));
      }
    }
    this.reset();
  }

  reset() {
    this.net.reset(1);
    this.H = 0; this.x = 0; this.y = 0; this.n = 0;
    this.lastT = 0; this.lastPhi = 0; this.lastSpeed = 1;
  }

  /** BodyVelocityInput.apply: drive the self-motion entries (LNO, SpsP) from |v| and phi. */
  _selfMotion(speed, phi) {
    const { gain, baseline } = this.data.bvi;
    for (const [g, cells] of Object.entries(this.routes)) {
      const p = (this.data.pref_deg[g] * Math.PI) / 180;
      const d = Math.max(0, Math.cos(phi - p)) * speed;
      for (const i of cells) this.net.clampVal[i] = Math.max(0, baseline + gain * d);
    }
  }

  /** Augmentation.apply: the added input, written into net.drive before the step. */
  _added(speed, phi) {
    if (!this.mechs.size) return;
    const net = this.net, r = net.r, v = this.cfg, drive = net.drive;
    if (this.mechs.has(1)) {
      let gate;
      if (this.variant === 'N') {
        const H = net.readHeading().theta;
        gate = this.pfn.map((i) => 1 + Math.cos(H - net.phase[i]));
      } else {
        gate = this.Wpe.map((row) => {
          let s = 0;
          for (let k = 0; k < row.length; k++) s += row[k] * r[net.epg[k]];
          return s / v.scales.epg_gate;
        });
      }
      let j = 0;
      this.groups.forEach((cells, g) => {
        const A = speed * Math.max(0, Math.cos(phi - this.pref[g]));
        for (const i of cells) { drive[i] += v.m * A * gate[j]; j++; }
      });
    }
    if (this.mechs.has(2)) {
      const add = new Float64Array(this.hdb.length);
      for (const [a, b, w] of this.M) add[a] += w * r[this.pfn[b]];
      this.hdb.forEach((i, a) => { drive[i] += add[a]; });
    }
    if (this.mechs.has(3)) {
      let g, t;
      if (this.variant === 'N') {
        const G = popAngle(r, this.fc2, this.fc2Phase).angle;
        const T = popAngle(r, this.hdb, this.psi).angle;
        if (G === null || T === null) return;
        const Th = T - v.c_T;
        g = this.gamma.map((gm) => Math.cos(G - gm));
        t = this.gamma.map((gm, a) => Math.cos(Th - gm - this.delta[a]));
      } else {
        const fc = this.Wpf.map((row) => {
          let s = 0;
          for (let k = 0; k < row.length; k++) s += row[k] * r[this.fc2[k]];
          return s;
        });
        g = new Array(fc.length);
        for (const s of [-1, 1]) {
          const idx = this.side.map((x, a) => (x === s ? a : -1)).filter((a) => a >= 0);
          const mean = idx.reduce((acc, a) => acc + fc[a], 0) / idx.length;
          for (const a of idx) g[a] = (fc[a] - mean) / v.scales.fc2_in;
        }
        t = this.PT.map((row) => {
          let s = 0;
          for (let k = 0; k < row.length; k++) s += row[k] * r[this.hdb[k]];
          return s / v.scales.hdb_proj;
        });
      }
      this.pfl3.forEach((i, a) => { const x = g[a] + t[a]; drive[i] += v.lam * x * x; });
    }
  }

  /**
   * One 20 ms step of the closed loop (scripts/servo_augmented.py closed_loop):
   * landmark at the current heading, goal into FC2, self-motion, added circuit, network step,
   * then the body turns by K * u * dt. Returns |T - G| in degrees (null when not moving).
   */
  step(G, phi, speed) {
    const net = this.net, d = this.data;
    net.clearDrive();
    net.setVisualScene(this.H, 1.0);
    net.injectGoal(G, d.goal_gain);
    this._selfMotion(speed, phi);
    this._added(speed, phi);
    net.step(d.dt);
    const T = this.H + phi;
    this.lastT = T; this.lastPhi = phi; this.lastSpeed = speed;
    this.x += speed * Math.cos(T) * d.dt;
    this.y += speed * Math.sin(T) * d.dt;
    this.H = wrap(this.H + d.K_plant * net.readSteering().turn * d.dt);
    this.n++;
    return speed > 1e-8 ? Math.abs(wrap(T - G)) * 180 / Math.PI : null;
  }

  /** What the page shows about the inside of the circuit. */
  snapshot() {
    const net = this.net, r = net.r;
    const s = net.readSteering();
    const h = popAngle(r, this.hdb, this.psi);
    const off = this.variant ? this.cfg.c_T : 0;
    return {
      pfn: this.groups.map((cells) => cells.reduce((a, i) => a + r[i], 0) / cells.length),
      hdb: this.hdb.map((i) => r[i]),
      hdbAngle: h.angle === null ? null : wrap(h.angle - off),
      hdbMag: h.mag,
      left: s.left / this.data.cells.pfl3_left.length,
      right: s.right / this.data.cells.pfl3_right.length,
      u: s.turn,
      heading: net.readHeading().theta,
    };
  }
}

/** Run one reference trial without drawing; returns the per-step errors after settling. */
export function runTrial(fly, data, trial) {
  const G = (data.reference.goals_deg[trial] * Math.PI) / 180;
  const track = data.reference.tracks[trial];
  fly.reset();
  const errs = [];
  for (let i = 0; i < track.length; i++) {
    const e = fly.step(G, track[i], 1.0);
    if (i >= data.settle_steps) errs.push(e);
  }
  return errs;
}

export function median(a) {
  if (!a.length) return null;
  const s = [...a].sort((x, y) => x - y), m = s.length >> 1;
  return s.length % 2 ? s[m] : 0.5 * (s[m - 1] + s[m]);
}
