#!/usr/bin/env python3
"""Paper B, section 7: supply the two missing multiplications as an added circuit and test it.

  python3 scripts/servo_augmented.py            # full run (about 20-40 minutes)
  python3 scripts/servo_augmented.py --quick    # coarse grids, a few trials (smoke test)
  -> data/servo_augmented.json

Why
---
Paper B shows that the plain circuit is a heading servo and that a travel servo
needs two bilinear operations: a multiplication that adds angles in the front
stage (PFN -> hDeltaB builds T = H + phi) and a square that subtracts them in
the comparator (PFL3 compares G with T). Here we add exactly those operations to
the connectome model, calibrate them once, and then measure what they buy and
how robust it is. Nothing in the base model changes: with every mechanism off
the dynamics are bit-identical (tests/test_servo_augmented.py).

Two variants
------------
N (decoded): the added terms use angles decoded from population activity:
   H^ from the EPG population vector, G^ from FC2, T^ from hDeltaB.
C (circuit): no decoding. The multiplication gates each PFN cell's own EPG
   synaptic input; the comparator squares the sum of each PFL3 cell's own FC2
   synaptic input and a newly declared hDeltaB -> PFL3 cosine projection.

Mechanisms (both variants)
--------------------------
(1) PFN multiplication. PFN cell j of group g (PFNd/PFNv x PB side) receives
      m * A_g * gate_j,     A_g = |v| [cos(phi - p_g)]_+,  p_g = +45, -45, -135, +135 deg
    N: gate_j = 1 + cos(H^ - alpha_j)       C: gate_j = EPG input to j / its calibrated maximum
    Control 'additive': m * (A_g + gate_j) / 2 (theory predicts no rotation).
(2) hDeltaB alignment. hDeltaB cell i receives q * sum_j M_ij r_j with M >= 0 and
    **only on PFN -> hDeltaB pairs that exist in the connectome**. M is fitted once
    by non-negative least squares on the training grid so that the added input
    follows [cos(psi_i - (theta + phi))]_+ * |v|.
    Controls: 'shuffled' (the fitted weights on randomly re-drawn existing pairs),
    'random' (degree-matched random PFN -> hDeltaB pairs, refitted).
(3) PFL3 squaring comparator. PFL3 cell i (phase gamma_i, side s) receives
      lam * [g_i + t_i]^2 ,   delta_L = +90 deg, delta_R = -90 deg
    N: g_i = cos(G^ - gamma_i),  t_i = cos(T^ - gamma_i - delta_s)
    C: g_i = (FC2 input to i - mean over its side) / scale  (carries the measured
       +/-73 deg offset o_s),  t_i = sum_k r_k cos(psi_k - c_T - gamma_i - o_s - delta_s) / scale
    Controls: 'linear' (no square), 'flipped' (delta_L and delta_R swapped).

Calibration (training grid only, then frozen; see calibrate())
-------------------------------------------------------------
m, q, lam and the native PFN scaling nu_P are chosen from a small fixed set by
the training-grid criteria below. c_T (the hDeltaB frame offset) and the
circuit-variant scales are measured on the training grid. After calibrate()
returns, no parameter is changed; evaluation grids are disjoint (held-out grid
is shifted by 22.5 deg) and the closed-loop tasks use the seeds of
scripts/bearing_vs_homing.py.

Pre-registered criteria (fixed before any evaluation)
----------------------------------------------------
E1 front stage   : a, b in [0.8, 1.2], residual R >= 0.8, amplitude R^2 >= 0.9 (coord_transform.CRITERIA)
E2 back stage    : R^2 >= 0.5, rho > 0.8 and K_T > 0 (paper B section 3.4)
E3 closed loop   : median travel error with sideslip <= 20 deg, and without sideslip
                   no worse than the plain circuit's CI upper bound (18.0 deg)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import numpy as np
from scipy.optimize import nnls

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from cxnet_np import CXNetworkNP                                    # noqa: E402
from coord_transform import (BodyVelocityInput, hdb_readout, circular_fit,  # noqa: E402
                             amplitude_fit, CRITERIA as FRONT_CRITERIA, PREF_DEG)
from servo_identify import fit_servo                                # noqa: E402
from bearing_vs_homing import sideslip_track, run_synthetic, boot_median  # noqa: E402

MODEL = os.path.join(ROOT, 'model', 'flybrain-cx.gguf')
OUT = os.path.join(ROOT, 'data', 'servo_augmented.json')
DT = 0.02
TAU = 2 * np.pi
K_PLANT = 2.6
SEED = 20260919
GROUPS = ('PFNd-L', 'PFNd-R', 'PFNv-L', 'PFNv-R')
DELTA = {-1: np.radians(+90.0), +1: np.radians(-90.0)}      # left, right

PREREG = {
    'front': FRONT_CRITERIA,
    'back': {'r2_min': 0.5, 'rho_min': 0.8, 'K_T_positive': True},
    'closed_loop': {'sideslip_max_deg': 20.0, 'no_sideslip_max_deg': 18.0},
}


def wrap(a):
    return (a + np.pi) % TAU - np.pi


def base_type(t):
    return re.sub(r'_.*', '', t)


def pop_angle(r, phase):
    x = float((r * np.cos(phase)).sum())
    y = float((r * np.sin(phase)).sum())
    return (np.arctan2(y, x) if np.hypot(x, y) > 1e-12 else None), float(np.hypot(x, y))


# ------------------------------------------------------------------ the added circuit
class Augmentation:
    """Adds mechanisms (1)(2)(3) to a CXNetworkNP by writing into net.drive each step.

    Call apply(speed, phi) after net.clear_drive() and the native inputs, and before
    net.step(). With mechs=() it writes nothing, so the base dynamics are unchanged.
    """

    def __init__(self, net, variant='N', mechs=(1, 2, 3), m=0.1, q=1.0, lam=0.05,
                 nu_P=1.0, nu_B=1.0, M=None, c_T=0.0, scales=None, mech1='mult',
                 mech2='fitted', mech3='square', seed=0):
        assert variant in ('N', 'C')
        self.net, self.variant, self.mechs = net, variant, set(mechs)
        self.m, self.q, self.lam, self.c_T = m, q, lam, c_T
        self.mech1, self.mech2, self.mech3 = mech1, mech2, mech3
        self.scales = dict(scales or {})
        T = [net.type_of(i) for i in range(net.N)]

        # PFN groups (PFNd/PFNv x PB side), as in scripts/pfn_basis.py
        self.group_idx = {}
        for g in GROUPS:
            typ, side = g.split('-')
            s = -1 if side == 'L' else +1
            self.group_idx[g] = np.array([i for i in range(net.N) if base_type(T[i]) == typ
                                          and net.pb_side[i] == s and net.phase[i] > -8],
                                         dtype=np.int64)
        self.pfn = np.concatenate([self.group_idx[g] for g in GROUPS])
        self.pref = {g: np.radians(PREF_DEG[g]) for g in GROUPS}
        if nu_P != 1.0 and 1 in self.mechs:
            net.row_gain[self.pfn] *= nu_P

        # EPG input to each PFN cell (circuit variant of the gate)
        W = net.W.tocsr()
        self.W_pfn_epg = W[self.pfn][:, net.epg].toarray().clip(min=0)

        # hDeltaB readout cells and their connectivity-derived phases
        self.hdb, self.psi = hdb_readout(net)
        self.allowed = (W[self.hdb][:, self.pfn].toarray() != 0)       # existing pairs
        if nu_B != 1.0 and 2 in self.mechs:
            net.row_gain[self.hdb] *= nu_B
        self.M = M if M is not None else np.zeros((len(self.hdb), len(self.pfn)))

        # PFL3 cells, sides, phases; FC2 cells and their input to PFL3
        self.pfl3 = np.concatenate([net.pfl3_l, net.pfl3_r])
        self.side = np.array([-1] * len(net.pfl3_l) + [+1] * len(net.pfl3_r))
        self.gamma = net.phase[self.pfl3]
        self.delta = np.array([DELTA[s] for s in self.side])
        if mech3 == 'flipped':
            self.delta = -self.delta
        self.fc2 = net._by_type_prefix(['FC2'])
        self.fc2_phase = net.phase[self.fc2]
        self.W_pfl3_fc2 = W[self.pfl3][:, self.fc2].toarray().clip(min=0)
        # measured FC2 -> PFL3 offsets (cx.pfl3_offset_left/right in the model file)
        self.o = np.where(self.side < 0, float(net.kv['cx.pfl3_offset_left']),
                          float(net.kv['cx.pfl3_offset_right']))
        # newly declared hDeltaB -> PFL3 cosine projection (circuit variant only)
        self.P_T = np.cos(self.psi[None, :] - c_T - self.gamma[:, None]
                          - self.o[:, None] - self.delta[:, None])
        rng = np.random.default_rng(seed)
        if 2 in self.mechs and mech2 == 'shuffled':
            self.M = self._shuffle_pairs(self.M, rng)

    # -- helpers
    def _shuffle_pairs(self, M, rng):
        """Move every fitted weight to a random existing pair (same number, same values)."""
        vals = M[M > 0]
        pairs = np.argwhere(self.allowed)
        pick = pairs[rng.choice(len(pairs), size=len(vals), replace=False)]
        out = np.zeros_like(M)
        out[pick[:, 0], pick[:, 1]] = vals
        return out

    def gate(self, H_hat):
        net = self.net
        if self.variant == 'N':
            alpha = net.phase[self.pfn]
            return 1.0 + np.cos(H_hat - alpha)
        e = self.W_pfn_epg @ net.r[net.epg]
        return e / self.scales.get('epg_gate', 1.0)

    def velocity(self, speed, phi):
        return np.concatenate([np.full(len(self.group_idx[g]),
                                       speed * max(0.0, np.cos(phi - self.pref[g])))
                               for g in GROUPS])

    def T_hat(self):
        a, _ = pop_angle(self.net.r[self.hdb], self.psi)
        return None if a is None else a - self.c_T

    def G_hat(self):
        a, _ = pop_angle(self.net.r[self.fc2], self.fc2_phase)
        return a

    # -- the added input
    def apply(self, speed, phi):
        net = self.net
        if not self.mechs:
            return
        if 1 in self.mechs:
            H_hat = net.read_heading()['theta'] if self.variant == 'N' else None
            A = self.velocity(speed, phi)
            gt = self.gate(H_hat)
            if self.mech1 == 'additive':
                add = self.m * 0.5 * (A + gt)
            else:
                add = self.m * A * gt
            net.drive[self.pfn] += add
        if 2 in self.mechs:
            net.drive[self.hdb] += self.q * (self.M @ net.r[self.pfn])
        if 3 in self.mechs:
            if self.variant == 'N':
                G, Tt = self.G_hat(), self.T_hat()
                if G is None or Tt is None:
                    return
                g = np.cos(G - self.gamma)
                t = np.cos(Tt - self.gamma - self.delta)
            else:
                fc = self.W_pfl3_fc2 @ net.r[self.fc2]
                g = np.empty_like(fc)
                for s in (-1, 1):
                    sel = self.side == s
                    g[sel] = fc[sel] - fc[sel].mean()
                g /= self.scales.get('fc2_in', 1.0)
                t = (self.P_T @ net.r[self.hdb]) / self.scales.get('hdb_proj', 1.0)
            x = g + t
            net.drive[self.pfl3] += self.lam * (x * x if self.mech3 != 'linear' else x)


# ------------------------------------------------------------------ building blocks
def make_net(goal_types=('FC2',)):
    net = CXNetworkNP(MODEL)
    net.set_goal_types(list(goal_types))
    return net


def settle(net, aug, bvi, theta, phi, speed, goal=None, steps=300):
    net.reset(1)
    for _ in range(steps):
        net.clear_drive()
        net.set_visual_scene(theta, 1.0)
        if goal is not None:
            net.inject_goal(goal, 0.3)
        bvi.apply(net.clamp_val, speed, phi)
        aug.apply(speed, phi)
        net.step(DT)


def front_sweep(aug, bvi, thetas, phis, speeds=(0.5, 1.0), steps=300, record=False):
    """hDeltaB population vector across (theta, phi, speed). Same readout as coord_transform."""
    net = aug.net
    rows, recs = [], []
    for td in thetas:
        for pd in phis:
            for sp in speeds:
                settle(net, aug, bvi, np.radians(td), np.radians(pd), sp, steps=steps)
                a = net.r[aug.hdb]
                psi, mag = pop_angle(a, aug.psi)
                rows.append({'theta': td, 'phi': pd, 'speed': sp, 'psi': psi, 'mag': mag,
                             'mean_rate': float(a.mean())})
                if record:
                    net.clear_drive()
                    recs.append({'theta': td, 'phi': pd, 'speed': sp,
                                 'pfn': net.r[aug.pfn].copy(),
                                 'hdb_syn': native_syn(net, aug.hdb)})
    return rows, recs


def front_verdict(fit, amp):
    if fit is None:
        return {'pass': False, 'why': 'readout silent'}
    c = FRONT_CRITERIA
    checks = {'heading_gain': c['heading_gain'][0] <= fit['heading_gain'] <= c['heading_gain'][1],
              'travel_gain': c['travel_gain'][0] <= fit['travel_gain'] <= c['travel_gain'][1],
              'residual_R': fit['residual_R'] >= c['residual_R_min'],
              'amplitude_r2': bool(amp and amp['r2'] is not None
                                   and amp['r2'] >= c['amplitude_r2_min'])}
    return {'pass': all(checks.values()), 'checks': checks}


def native_syn(net, idx):
    """The synaptic input the base model gives cells idx right now (no shunting)."""
    gi = net.global_inhibition * net.r.mean()
    return ((net.W[idx] @ net.r) * net.row_gain[idx] * net.excitability
            + net.ext_coef[idx] * net.external_drive + net.bias[idx] + net.drive[idx] - gi)


def fit_M(aug, recs, amp=0.3, q=1.0, max_nonzero=None):
    """NNLS per hDeltaB cell, only over existing PFN -> hDeltaB pairs.

    The target is an output: hDeltaB rate r* = amp * speed * [cos(psi_i - (theta + phi))]_+.
    Through the inverse of the activation, f^-1(r) = threshold + r / (1 - r), this gives the
    total input the cell needs; subtracting the native input it already receives (recorded
    with the added circuit off) leaves what q * M r_PFN must supply. Negative remainders
    cannot be supplied by M >= 0 and are clipped to zero.
    """
    thr = aug.net.threshold
    R = np.array([rec['pfn'] for rec in recs])                      # conditions x pfn
    NAT = np.array([rec['hdb_syn'] for rec in recs])                # conditions x hdb
    tgt_T = np.radians([rec['theta'] + rec['phi'] for rec in recs])
    sp = np.array([rec['speed'] for rec in recs])
    M = np.zeros((len(aug.hdb), len(aug.pfn)))
    for i in range(len(aug.hdb)):
        cols = np.where(aug.allowed[i])[0]
        if len(cols) == 0:
            continue
        rstar = np.clip(amp * sp * np.maximum(0.0, np.cos(aug.psi[i] - tgt_T)), 0, 0.9)
        need = np.where(rstar > 0, thr + rstar / (1 - rstar), 0.0) - NAT[:, i]
        need = np.maximum(need, 0.0) / q
        w, _ = nnls(R[:, cols], need)
        M[i, cols] = w
    if max_nonzero is not None and (M > 0).sum() > max_nonzero:
        cut = np.sort(M[M > 0])[::-1][max_nonzero - 1]
        M[M < cut] = 0.0
    return M


def back_sweep(aug, bvi, thetas, goals, phis, steps=300, speed=1.0):
    net = aug.net
    rows = []
    for t in thetas:
        for g in goals:
            for p in phis:
                settle(net, aug, bvi, np.radians(t), np.radians(p), speed,
                       goal=np.radians(g), steps=steps)
                s = net.read_steering()
                rows.append({'theta': t, 'goal': g, 'phi': p, 'u': float(s['turn'])})
    return rows


def back_verdict(fit):
    if fit is None:
        return {'pass': False}
    c = PREREG['back']
    checks = {'r2': fit['r2'] >= c['r2_min'], 'rho': (fit['rho'] or 0) > c['rho_min'],
              'K_T_positive': fit['K_T'] > 0}
    return {'pass': all(checks.values()), 'checks': checks}


# ------------------------------------------------------------------ closed loop
def closed_loop(aug, bvi, goal, phi_track, K=K_PLANT, settle_steps=150, speed_track=None,
                delay_steps=0, noise=0.0, H0=0.0, wind_beta=None, wind_dir=0.0, rng=None):
    """One trial. Returns |T - G| (deg) after settle_steps.

    delay_steps delays what the circuit sees (landmark and self-motion) by that many
    steps. noise adds Gaussian noise (std, in units of speed) to the self-motion drive.
    wind_beta, if set, replaces phi_track by a world-fixed wind phi = beta sin(w - H).
    """
    net = aug.net
    net.reset(1)
    H = H0
    buf = []
    errs = []
    for i in range(len(phi_track)):
        phi = phi_track[i] if wind_beta is None else wind_beta * np.sin(wind_dir - H)
        sp = 1.0 if speed_track is None else speed_track[i]
        buf.append((H, phi, sp))
        Hs, phis, sps = buf[max(0, len(buf) - 1 - delay_steps)]
        if noise > 0 and rng is not None:
            sps_seen = max(0.0, sps + noise * rng.standard_normal())
            phis_seen = phis + noise * rng.standard_normal()
        else:
            sps_seen, phis_seen = sps, phis
        net.clear_drive()
        net.set_visual_scene(Hs, 1.0)
        net.inject_goal(goal, 0.3)
        bvi.apply(net.clamp_val, sps_seen, phis_seen)
        aug.apply(sps_seen, phis_seen)
        net.step(DT)
        T = H + phi
        H = wrap(H + K * net.read_steering()['turn'] * DT)
        if i >= settle_steps:
            errs.append(abs(np.degrees(wrap(T - goal))))
    return errs


def speed_track_ou(n, rng, mean=1.0, sd=0.25, tau_s=2.0, lo=0.5, hi=1.5):
    """Walking speed as an OU process around 1.0 (sd 0.25, tau 2 s, clipped to [0.5, 1.5])."""
    a = np.exp(-DT / tau_s)
    x, out = 0.0, np.empty(n)
    for i in range(n):
        x = a * x + np.sqrt(1 - a * a) * sd * rng.standard_normal()
        out[i] = np.clip(mean + x, lo, hi)
    return out


def task(aug, bvi, trials=12, steps=900, sigma=55.0, seed=SEED, goal_shift=0.0,
         random_H0=False, speed_ou=False, **kw):
    goals = [np.radians(g + goal_shift) for g in (30, 90, 150, 210, 270, 330)]
    rng = np.random.default_rng(seed)
    tracks = [sideslip_track(steps, DT, sigma, 4.0, 60.0, rng) for _ in range(trials)]
    rng2 = np.random.default_rng(seed + 1)
    errs = []
    per_trial = []
    for k, tr in enumerate(tracks):
        H0 = rng2.uniform(-np.pi, np.pi) if random_H0 else 0.0
        sp = speed_track_ou(steps, rng2) if speed_ou else None
        e = closed_loop(aug, bvi, goals[k % len(goals)], tr, H0=H0, speed_track=sp,
                        rng=rng2, **kw)
        errs += e
        per_trial.append(float(np.median(e)))
    med, ci = boot_median(errs)
    return {'median_deg': med, 'ci95': ci, 'p90_deg': float(np.percentile(errs, 90)),
            'per_trial_median_deg': per_trial, 'n_samples': len(errs)}


def law_only(KH, KT, c, sigma=55.0):
    """The fitted control law alone on the same task (scripts/servo_theory.py run_mixed)."""
    import servo_theory as st
    goals = [np.radians(g) for g in st.GOALS_DEG]
    errs = []
    for k, tr in enumerate(st.tracks(sigma)):
        errs += st.run_mixed(KH, KT, c, goals[k % len(goals)], tr)
    return float(np.median(errs))


def budget(fit, obs_ss, obs_no):
    """Paper B section 6.2 decomposition applied to an augmented circuit."""
    law_ss = law_only(fit['K_H'], fit['K_T'], fit['c'], 55.0)
    law_no = law_only(fit['K_H'], fit['K_T'], fit['c'], 0.0)
    floor = float(np.sqrt(max(obs_no ** 2 - law_no ** 2, 0.0)))
    return {'law_sideslip_deg': law_ss, 'law_no_sideslip_deg': law_no, 'floor_deg': floor,
            'rss_prediction_deg': float(np.hypot(law_ss, floor)),
            'effective_loop_gain': K_PLANT * (fit['K_H'] + fit['K_T']),
            'static_component_deg': float(abs(fit['K_H'] / (fit['K_H'] + fit['K_T'])) * 40.0)
            if fit['K_H'] + fit['K_T'] != 0 else None}


# ------------------------------------------------------------------ calibration
TRAIN_THETA = list(range(0, 360, 45))
TRAIN_PHI = list(range(0, 360, 45))
HELD_THETA = [t + 22.5 for t in TRAIN_THETA]
HELD_PHI = [p + 22.5 for p in TRAIN_PHI]
CAL_SPEEDS = (0.25, 0.5, 0.75, 1.0)
CAL_M = {'N': (0.05, 0.1, 0.2), 'C': (0.1, 0.2, 0.4)}
CAL_NU_P = (0.25, 0.5)
CAL_LAMBDA = (0.02, 0.05, 0.1, 0.2, 0.4)
NU_B, M_AMP = 0.5, 0.2
# back-stage training grid: disjoint from the evaluation grid (offset by 45 / 22.5 deg)
TRAIN_BACK = ([45, 135, 225, 315], [g + 22.5 for g in range(0, 360, 45)],
              [p + 22.5 for p in range(0, 360, 45)])
EVAL_BACK = ([0, 90, 180, 270], list(range(0, 360, 45)), list(range(0, 360, 45)))


def epg_gate_scale(quick=False):
    """Half the largest EPG input any PFN cell receives across a heading sweep."""
    net = make_net(); bvi = BodyVelocityInput(net)
    aug = Augmentation(net, 'C', mechs=())
    vals = []
    for t in range(0, 360, 90 if quick else 45):
        settle(net, aug, bvi, np.radians(t), 0.0, 0.0, steps=200)
        vals.append(aug.W_pfn_epg @ net.r[net.epg])
    return 0.5 * float(np.max(vals))


def build_front(variant, m, nu_P, scales, allowed=None, quick=False):
    """Fresh network with mechanisms (1)(2); M fitted on the training grid."""
    net = make_net(); bvi = BodyVelocityInput(net)
    aug = Augmentation(net, variant, mechs=(1, 2), m=m, nu_P=nu_P, nu_B=NU_B, scales=scales)
    if allowed is not None:
        aug.allowed = allowed
    th = TRAIN_THETA[::2] if quick else TRAIN_THETA
    ph = TRAIN_PHI[::2] if quick else TRAIN_PHI
    _, recs = front_sweep(aug, bvi, th, ph, speeds=CAL_SPEEDS, record=True)
    aug.M = fit_M(aug, recs, amp=M_AMP)
    rows, _ = front_sweep(aug, bvi, th, ph)
    return net, bvi, aug, rows


def circuit_scales(net, bvi, aug_full):
    G, Tt = [], []
    for t in (0, 90, 180, 270):
        for g in (0, 90, 180, 270):
            settle(net, aug_full, bvi, np.radians(t), np.radians(g), 1.0, goal=np.radians(g), steps=200)
            fc = aug_full.W_pfl3_fc2 @ net.r[aug_full.fc2]
            gg = np.concatenate([fc[aug_full.side == s] - fc[aug_full.side == s].mean() for s in (-1, 1)])
            G.append(np.abs(gg).max())
            Tt.append(np.abs(aug_full.P_T @ net.r[aug_full.hdb]).max())
    return float(np.median(G)), float(np.median(Tt))


def calibrate(variant, quick=False, log=print):
    """Choose m, nu_P (front) and lambda (back) on training grids only. Returns a frozen config."""
    scales = {}
    if variant == 'C':
        scales['epg_gate'] = epg_gate_scale(quick)
    cands = []
    for m in CAL_M[variant][:2] if quick else CAL_M[variant]:
        for nu_P in CAL_NU_P[:1] if quick else CAL_NU_P:
            net, bvi, aug, rows = build_front(variant, m, nu_P, scales, quick=quick)
            fit, amp = circular_fit(rows), amplitude_fit(rows)
            ok = bool(fit and all(FRONT_CRITERIA[k][0] <= fit[k] <= FRONT_CRITERIA[k][1]
                                  for k in ('heading_gain', 'travel_gain')))
            score = (ok, fit['residual_R'] if fit else 0, (amp or {}).get('r2') or -9)
            log(f'  [{variant}] m={m} nu_P={nu_P}: a/b in band {ok}, '
                f'R {score[1]:.3f}, amp R2 {score[2]:.3f}')
            cands.append({'m': m, 'nu_P': nu_P, 'score': score,
                          'offset_deg': fit['offset_deg'] if fit else 0.0})
    best = max(cands, key=lambda c: c['score'])
    cfg = {'variant': variant, 'm': best['m'], 'nu_P': best['nu_P'], 'nu_B': NU_B,
           'M_amp': M_AMP, 'c_T_deg': best['offset_deg'], 'scales': scales,
           'front_candidates': [{k: v for k, v in c.items()} for c in cands]}
    net, bvi, aug, _ = build_front(variant, cfg['m'], cfg['nu_P'], scales, quick=quick)
    cfg['M'] = aug.M
    if variant == 'C':
        probe = Augmentation(net, 'C', mechs=(1, 2, 3), m=cfg['m'], M=aug.M,
                             c_T=np.radians(cfg['c_T_deg']), lam=0.0, scales=scales)
        scales['fc2_in'], scales['hdb_proj'] = circuit_scales(net, bvi, probe)
    th, gs, ps = TRAIN_BACK
    if quick:
        th, gs, ps = th[:2], gs[::2], ps[::2]
    lam_rows = []
    for lam in CAL_LAMBDA:
        full = Augmentation(net, variant, mechs=(1, 2, 3), m=cfg['m'], M=aug.M,
                            c_T=np.radians(cfg['c_T_deg']), lam=lam, scales=scales)
        fit = fit_servo(back_sweep(full, bvi, th, gs, ps))
        good = fit['r2'] >= 0.5 and (fit['rho'] or 0) > 0.9 and fit['K_T'] > 0
        lam_rows.append({'lam': lam, 'K_H': fit['K_H'], 'K_T': fit['K_T'], 'rho': fit['rho'],
                         'r2': fit['r2'], 'chosen_rule_met': good})
        log(f'  [{variant}] lambda={lam}: K_H {fit["K_H"]:+.3f} K_T {fit["K_T"]:+.3f} '
            f'rho {fit["rho"]:.3f} R2 {fit["r2"]:.3f}')
        if good:
            break
    met = [r for r in lam_rows if r['chosen_rule_met']]
    cfg['lam'] = (met[0] if met else max(lam_rows, key=lambda r: r['rho']))['lam']
    cfg['lambda_candidates'] = lam_rows
    cfg['rule'] = ('front: maximise (a and b in band, residual R, amplitude R2) on the training grid; '
                   'back: smallest lambda with rho > 0.9, K_T > 0, R2 >= 0.5 on the training back grid')
    return cfg


def instantiate(cfg, mechs=(1, 2, 3), goal_types=('FC2',), **over):
    """A fresh network carrying the frozen configuration (optionally with overrides)."""
    net = make_net(goal_types); bvi = BodyVelocityInput(net)
    kw = dict(m=cfg['m'], nu_P=cfg['nu_P'], nu_B=cfg['nu_B'], M=cfg['M'],
              c_T=np.radians(cfg['c_T_deg']), lam=cfg['lam'], scales=cfg['scales'])
    kw.update(over)
    aug = Augmentation(net, cfg['variant'], mechs=mechs, **kw)
    return net, bvi, aug


def random_pairs_like(aug, rng):
    """Degree-matched random PFN -> hDeltaB pairs (same count per hDeltaB cell)."""
    out = np.zeros_like(aug.allowed)
    for i in range(aug.allowed.shape[0]):
        k = int(aug.allowed[i].sum())
        out[i, rng.choice(aug.allowed.shape[1], size=k, replace=False)] = True
    return out


# ------------------------------------------------------------------ evaluation
def evaluate(cfg, quick=False, log=print):
    v = cfg['variant']
    res = {}
    trials = 4 if quick else 12
    # E1 front stage (mechanisms 1, 2 only; the comparator does not feed back into hDeltaB)
    net, bvi, aug = instantiate(cfg, mechs=(1, 2))
    for name, th, ph in (('train', TRAIN_THETA, TRAIN_PHI), ('held_out', HELD_THETA, HELD_PHI)):
        if quick:
            th, ph = th[::2], ph[::2]
        rows, _ = front_sweep(aug, bvi, th, ph)
        fit, amp = circular_fit(rows), amplitude_fit(rows)
        res[f'front_{name}'] = {'fit': fit, 'amplitude': amp, 'verdict': front_verdict(fit, amp)}
        log(f'  [{v}] E1 {name}: a {fit["heading_gain"]:+.2f} b {fit["travel_gain"]:+.2f} '
            f'R {fit["residual_R"]:.3f} amp R2 {amp["r2"]:.3f}')
    # E2 back stage
    net, bvi, aug = instantiate(cfg)
    th, gs, ps = EVAL_BACK
    if quick:
        th, gs, ps = th[:2], gs[::2], ps[::2]
    fit = fit_servo(back_sweep(aug, bvi, th, gs, ps))
    res['back'] = {'fit': fit, 'verdict': back_verdict(fit)}
    log(f'  [{v}] E2: K_H {fit["K_H"]:+.3f} K_T {fit["K_T"]:+.3f} rho {fit["rho"]:.3f} R2 {fit["r2"]:.3f}')
    # E3 closed loop
    ss = task(aug, bvi, trials=trials, sigma=55.0)
    no = task(aug, bvi, trials=trials, sigma=0.0)
    c = PREREG['closed_loop']
    res['closed_loop'] = {'sideslip': ss, 'no_sideslip': no,
                          'pass': ss['median_deg'] <= c['sideslip_max_deg']
                          and no['median_deg'] <= c['no_sideslip_max_deg']}
    res['budget'] = budget(fit, ss['median_deg'], no['median_deg'])
    log(f'  [{v}] E3: sideslip {ss["median_deg"]:.1f} [{ss["ci95"][0]:.1f}, {ss["ci95"][1]:.1f}] '
        f'no sideslip {no["median_deg"]:.1f}  law {res["budget"]["law_sideslip_deg"]:.1f} '
        f'RSS {res["budget"]["rss_prediction_deg"]:.1f}')
    return res


def ablations_and_controls(cfg, quick=False, log=print):
    v = cfg['variant']
    trials = 4 if quick else 12
    th, gs, ps = EVAL_BACK
    if quick:
        th, gs, ps = th[:2], gs[::2], ps[::2]
    conds = {
        'without (1) multiplication': dict(mechs=(2, 3)),
        'without (2) alignment': dict(mechs=(1, 3)),
        'without (3) comparator': dict(mechs=(1, 2)),
        '(3) linear, no square': dict(mech3='linear'),
        '(3) delta swapped': dict(mech3='flipped'),
        '(1) additive, not multiplicative': dict(mech1='additive'),
        '(2) weights on shuffled pairs': dict(mech2='shuffled'),
    }
    out = {}
    for name, over in conds.items():
        mechs = over.pop('mechs', (1, 2, 3))
        net, bvi, aug = instantiate(cfg, mechs=mechs, **over)
        row = {}
        if 3 not in mechs or name.startswith(('(1)', '(2)', 'without (1)', 'without (2)')):
            fr, _ = front_sweep(aug, bvi, TRAIN_THETA[::2 if quick else 1], TRAIN_PHI[::2 if quick else 1])
            row['front'] = circular_fit(fr)
        fit = fit_servo(back_sweep(aug, bvi, th, gs, ps))
        row['back'] = fit
        row['sideslip'] = task(aug, bvi, trials=trials, sigma=55.0)
        out[name] = row
        f = row.get('front') or {}
        log(f'  [{v}] {name:34s} b {f.get("travel_gain", float("nan")):+.2f}  '
            f'K_H {fit["K_H"]:+.3f} K_T {fit["K_T"]:+.3f} rho {fit["rho"]:.3f}  '
            f'err {row["sideslip"]["median_deg"]:.1f}')
    # degree-matched random pairs, M refitted on them
    rng = np.random.default_rng(1)
    probe_net, _, probe = instantiate(cfg, mechs=(1, 2))
    allowed = random_pairs_like(probe, rng)
    net, bvi, aug, _ = build_front(v, cfg['m'], cfg['nu_P'], cfg['scales'], allowed=allowed, quick=quick)
    fr, _ = front_sweep(aug, bvi, TRAIN_THETA[::2 if quick else 1], TRAIN_PHI[::2 if quick else 1])
    full = Augmentation(net, v, mechs=(1, 2, 3), m=cfg['m'], M=aug.M,
                        c_T=np.radians(cfg['c_T_deg']), lam=cfg['lam'], scales=cfg['scales'])
    fit = fit_servo(back_sweep(full, bvi, th, gs, ps))
    out['(2) refitted on random pairs'] = {'front': circular_fit(fr), 'back': fit,
                                          'sideslip': task(full, bvi, trials=trials, sigma=55.0)}
    log(f'  [{v}] (2) refitted on random pairs     err '
        f'{out["(2) refitted on random pairs"]["sideslip"]["median_deg"]:.1f}')
    return out


def robustness(cfg, quick=False, log=print, base=False):
    """E6. base=True runs the same conditions on the plain circuit for reference."""
    trials = 4 if quick else 12
    mk = (lambda **o: instantiate(cfg, mechs=(), **o)) if base else (lambda **o: instantiate(cfg, **o))
    net, bvi, aug = mk()
    conds = [('reference', {})]
    conds += [(f'seed set {k}', {'seed': SEED + 1000 * k}) for k in range(1, 3 if quick else 5)]
    conds += [('goals shifted 15 deg', {'goal_shift': 15.0}),
              ('random initial heading', {'random_H0': True}),
              ('fluctuating speed', {'speed_ou': True}),
              ('self-motion noise 0.1', {'noise': 0.1}),
              ('self-motion noise 0.2', {'noise': 0.2}),
              ('delay 40 ms', {'delay_steps': 2}),
              ('delay 100 ms', {'delay_steps': 5}),
              ('loop gain K = 1.3', {'K': 1.3}),
              ('loop gain K = 5.2', {'K': 5.2}),
              ('world-fixed wind beta 0.3', {'wind_beta': 0.3, 'wind_dir': 1.0}),
              ('world-fixed wind beta 0.7', {'wind_beta': 0.7, 'wind_dir': 1.0}),
              ('world-fixed wind beta 1.3', {'wind_beta': 1.3, 'wind_dir': 1.0})]
    out = {}
    for name, kw in conds:
        r = task(aug, bvi, trials=trials, sigma=55.0, **kw)
        out[name] = r
        log(f'  [{"base" if base else cfg["variant"]}] {name:28s} {r["median_deg"]:6.1f} '
            f'[{r["ci95"][0]:.1f}, {r["ci95"][1]:.1f}]')
    if not base:
        for f in (0.5, 2.0):
            net, bvi, aug = instantiate(cfg, lam=cfg['lam'] * f)
            r = task(aug, bvi, trials=trials, sigma=55.0)
            out[f'lambda x{f:g}'] = r
            log(f'  [{cfg["variant"]}] lambda x{f:g}{"":20s} {r["median_deg"]:6.1f}')
    return out


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--out', default=OUT)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--variants', default='N,C')
    a = ap.parse_args()
    t0 = time.time()

    def log(msg):
        print(f'{time.time() - t0:7.0f}s {msg}', flush=True)

    out = {'prereg': PREREG, 'quick': a.quick,
           'note': 'Added circuit for paper B section 7. Parameters are calibrated on training '
                   'grids and frozen before evaluation. No number from any external report is used.',
           'variants': {}}
    log('=== plain circuit (every mechanism off), same code path')
    cfg0 = {'variant': 'N', 'm': 0, 'nu_P': 1, 'nu_B': 1, 'M': None, 'c_T_deg': 0, 'lam': 0,
            'scales': {}}
    net, bvi, aug = instantiate(cfg0, mechs=())
    th, gs, ps = EVAL_BACK if not a.quick else (EVAL_BACK[0][:2], EVAL_BACK[1][::2], EVAL_BACK[2][::2])
    fit0 = fit_servo(back_sweep(aug, bvi, th, gs, ps))
    trials = 4 if a.quick else 12
    out['plain'] = {'back': fit0, 'sideslip': task(aug, bvi, trials=trials, sigma=55.0),
                    'no_sideslip': task(aug, bvi, trials=trials, sigma=0.0)}
    log(f'  plain: K_H {fit0["K_H"]:+.3f} K_T {fit0["K_T"]:+.3f} '
        f'sideslip {out["plain"]["sideslip"]["median_deg"]:.2f} '
        f'no sideslip {out["plain"]["no_sideslip"]["median_deg"]:.2f}')
    out['plain_robustness'] = robustness(cfg0, a.quick, log, base=True)
    # ideal servos on the same task, for scale
    out['ideal'] = {'travel': law_only(0.0, 1.0, 0.0), 'heading': law_only(1.0, 0.0, 0.0)}

    for v in a.variants.split(','):
        log(f'=== variant {v}: calibration (training grids only)')
        cfg = calibrate(v, a.quick, log)
        log(f'  frozen: m={cfg["m"]} nu_P={cfg["nu_P"]} lambda={cfg["lam"]} '
            f'c_T={cfg["c_T_deg"]:.1f} deg, non-zero M {(cfg["M"] > 0).sum()}')
        rec = {'config': {k: val for k, val in cfg.items() if k != 'M'},
               'M_nonzero': int((cfg['M'] > 0).sum()),
               'M_pfn_cells': int(((cfg['M'] > 0).sum(0) > 0).sum()),
               'M_hdb_cells': int(((cfg['M'] > 0).sum(1) > 0).sum())}
        log(f'=== variant {v}: evaluation')
        rec['evaluation'] = evaluate(cfg, a.quick, log)
        log(f'=== variant {v}: ablations and controls')
        rec['ablations'] = ablations_and_controls(cfg, a.quick, log)
        log(f'=== variant {v}: robustness')
        rec['robustness'] = robustness(cfg, a.quick, log)
        out['variants'][v] = rec
        with open(a.out, 'w', encoding='utf-8') as fh:
            json.dump(jsonable(out), fh, indent=1)
    out['elapsed_s'] = time.time() - t0
    with open(a.out, 'w', encoding='utf-8') as fh:
        json.dump(jsonable(out), fh, indent=1)
    log(f'-> {a.out}')


if __name__ == '__main__':
    main()
