#!/usr/bin/env python3
"""Export paper B section 7's added circuit for the browser demo (web/servo.html).

  python3 scripts/export_servo_demo.py [-o web/servo_demo_data.json]     (about 10 minutes)

data/servo_augmented.json stores the frozen configuration of both variants but not the
alignment weights M. calibrate() in scripts/servo_augmented.py is deterministic, so this
script re-runs it, checks that the result is the configuration that was evaluated
(m, nu_P, lambda, c_T and the number of non-zero weights), and writes what the
JavaScript port (web/servo_augmented.js) needs:

  * per variant: m, nu_P, nu_B, lambda, c_T, the circuit-variant scales, M as sparse triplets
  * cell orders: PFN groups, the hDeltaB readout cells and their connectivity-derived
    phases, PFL3 left then right
  * the self-motion entry routes of BodyVelocityInput (LNO1/LNO2/SpsP, left/right)
  * the reference task: 12 OU sideslip tracks and goals (seed 20260919), and the per-trial
    medians measured in Python for the plain circuit and both variants
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import servo_augmented as sa                                        # noqa: E402
from bearing_vs_homing import sideslip_track                        # noqa: E402

OUT = os.path.join(ROOT, 'web', 'servo_demo_data.json')
TRIALS, STEPS, GOALS = 12, 900, (30, 90, 150, 210, 270, 330)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--out', default=OUT)
    a = ap.parse_args()
    with open(os.path.join(ROOT, 'data', 'servo_augmented.json'), encoding='utf-8') as f:
        saved = json.load(f)

    out = {'source': 'scripts/export_servo_demo.py from scripts/servo_augmented.py calibrate(); '
                     'checked against data/servo_augmented.json',
           'dt': sa.DT, 'K_plant': sa.K_PLANT, 'settle_steps': 150,
           'goal_gain': 0.3, 'bvi': {'gain': 0.08, 'baseline': 0.02},
           'pref_deg': {g: sa.PREF_DEG[g] for g in sa.GROUPS},
           'variants': {}}

    for v in ('N', 'C'):
        print(f'calibrating variant {v} (deterministic re-run)...', flush=True)
        cfg = sa.calibrate(v, quick=False, log=lambda *_: None)
        ref = saved['variants'][v]
        for k in ('m', 'nu_P', 'lam'):
            assert cfg[k] == ref['config'][k], (v, k, cfg[k], ref['config'][k])
        assert abs(cfg['c_T_deg'] - ref['config']['c_T_deg']) < 1e-9, (v, 'c_T')
        nz = int((cfg['M'] > 0).sum())
        assert nz == ref['M_nonzero'], (v, nz, ref['M_nonzero'])
        net, bvi, aug = sa.instantiate(cfg)
        rows, cols = np.nonzero(cfg['M'])
        out['variants'][v] = {
            'm': cfg['m'], 'nu_P': cfg['nu_P'], 'nu_B': cfg['nu_B'], 'lam': cfg['lam'],
            'c_T': float(np.radians(cfg['c_T_deg'])), 'scales': cfg['scales'],
            'M': [[int(i), int(j), float(cfg['M'][i, j])] for i, j in zip(rows, cols)],
            'M_nonzero': nz,
        }
        print(f'  ok: m={cfg["m"]} nu_P={cfg["nu_P"]} lambda={cfg["lam"]} non-zero M={nz}')

    # cell orders (identical for both variants)
    out['cells'] = {
        'groups': {g: aug.group_idx[g].tolist() for g in sa.GROUPS},
        'hdb': aug.hdb.tolist(), 'psi': aug.psi.tolist(),
        'pfl3_left': net.pfl3_l.tolist(), 'pfl3_right': net.pfl3_r.tolist(),
        'fc2': aug.fc2.tolist(),
        'pfl3_offset': {'left': float(net.kv['cx.pfl3_offset_left']),
                        'right': float(net.kv['cx.pfl3_offset_right'])},
    }
    out['bvi_routes'] = {
        'PFNd-R': [int(i) for i in np.concatenate([bvi.lno2_left, net.spsp_right])],
        'PFNd-L': [int(i) for i in np.concatenate([bvi.lno2_right, net.spsp_left])],
        'PFNv-R': [int(i) for i in bvi.lno1_left],
        'PFNv-L': [int(i) for i in bvi.lno1_right],
    }

    rng = np.random.default_rng(sa.SEED)
    tracks = [sideslip_track(STEPS, sa.DT, 55.0, 4.0, 60.0, rng) for _ in range(TRIALS)]
    out['reference'] = {
        'seed': sa.SEED, 'sigma_deg': 55.0, 'tau_s': 4.0, 'max_deg': 60.0,
        'goals_deg': [GOALS[k % len(GOALS)] for k in range(TRIALS)],
        'tracks': [[round(float(x), 7) for x in tr] for tr in tracks],
        'python_per_trial_median_deg': {
            'plain': saved['plain']['sideslip']['per_trial_median_deg'],
            'N': saved['variants']['N']['evaluation']['closed_loop']['sideslip']['per_trial_median_deg'],
            'C': saved['variants']['C']['evaluation']['closed_loop']['sideslip']['per_trial_median_deg']},
        'python_median_deg': {
            'plain': saved['plain']['sideslip']['median_deg'],
            'N': saved['variants']['N']['evaluation']['closed_loop']['sideslip']['median_deg'],
            'C': saved['variants']['C']['evaluation']['closed_loop']['sideslip']['median_deg']},
    }
    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, separators=(',', ':'))
    print(f'-> {a.out} ({os.path.getsize(a.out) / 1e3:.0f} kB)')


if __name__ == '__main__':
    main()
