#!/usr/bin/env python3
"""Figures and measurements for paper A (the basic navigation algorithm).

  python3 paper/scripts/fig_a.py            # measure + draw (about 2 minutes)
  python3 paper/scripts/fig_a.py --quick    # skip the 48-trial closed-loop replication

Outputs
  paper/data/paper_a_measurements.json      every number measured here
  paper/a_navigation/figures/figA*.png      English figures

Inputs are model/flybrain-cx.gguf (the Q8_0 model the browser runs),
data/cx_network.npz (the signed synapse counts it was built from) and
data/claims_audit_maze.json (the 48-trial closed-loop audit run with the
JavaScript engine, web/cxnet.js). The Python engine (scripts/cxnet_np.py)
matches the JavaScript one; the closed-loop replication below re-measures the
audit with it as an independent check.
"""

from __future__ import annotations

import argparse
import os
import re

import numpy as np
from matplotlib.patches import FancyBboxPatch, Circle

from common import (BLUE, DIM, GREEN, GRID, INK, MODEL, ORANGE, PURPLE, RED, TAU,
                    dense, load_json, load_npz, note, panel_title, phase_kernel, plt,
                    requantize, save, save_json, select, wrap)
from cxnet_np import CXNetworkNP
from gguf import GGUFReader

P = 'a_navigation'
DT, K_TURN, GOAL_GAIN = 0.02, 2.6, 0.3          # same as scripts/audit_claims.mjs
# A configuration "passes" when its median closed-loop |heading error| over the 48 audit
# trials is below 30 deg (chance is 90 deg). Fixed before the table was computed.
PASS_DEG = 30.0


# ---------------------------------------------------------------- measurements
def ring_embedding(D):
    """Spectral embedding of the two-hop EPG->EPG graph (scripts/extract_cx.py)."""
    epg = select(D['types'], 'EPG', 'EPGt', exact=True)
    A = np.abs(dense(D))
    A2 = A @ A
    M = A2[np.ix_(epg, epg)] + A[np.ix_(epg, epg)] * 50.0
    M = 0.5 * (M + M.T)
    d = M.sum(1)
    d[d == 0] = 1
    Mn = M / d[:, None]
    Mn = 0.5 * (Mn + Mn.T)
    vals, vecs = np.linalg.eigh(Mn)
    o = np.argsort(vals)[::-1]
    v1, v2 = vecs[:, o[1]], vecs[:, o[2]]
    theta = np.mod(np.arctan2(v2, v1), TAU)
    glom = []
    for nm in D['names'][epg]:
        m = re.search(r'_([LR])(\d)$', nm)
        glom.append(m.group(1) + m.group(2) if m else '?')
    dev = np.degrees(np.abs(wrap(theta - D['phase'][epg])))
    return {'idx': epg, 'theta': theta, 'v1': v1, 'v2': v2, 'glom': glom,
            'eig': vals[o[:6]].tolist(), 'max_dev_deg': float(dev.max())}


def glomerulus_order(ring):
    """Glomerulus labels sorted by their mean recovered angle."""
    by = {}
    for g, t in zip(ring['glom'], ring['theta']):
        by.setdefault(g, []).append(t)
    mean = {g: float(np.mod(np.arctan2(np.sin(v).mean(), np.cos(v).mean()), TAU))
            for g, v in by.items()}
    return sorted(mean, key=mean.get), mean


def pfl3_offsets(D):
    """Goal-input phase minus heading-input phase, per PFL3 cell (build_gguf / audit_claims)."""
    W = dense(D)
    ph, t = D['phase'], D['types']
    epg = select(t, 'EPG', exact=True)
    fc2 = select(t, 'FC2')
    pfl3 = select(t, 'PFL3', exact=True)

    def mean_phase(cell, src):
        src = src[ph[src] > -8]
        w = np.maximum(W[cell, src], 0)
        if w.sum() <= 0:
            return None
        return np.arctan2((w * np.sin(ph[src])).sum(), (w * np.cos(ph[src])).sum())

    out = {}
    for side, name in ((-1, 'left'), (+1, 'right')):
        ds = []
        for c in pfl3[D['side'][pfl3] == side]:
            h, g = mean_phase(c, epg), mean_phase(c, fc2)
            if h is not None and g is not None:
                ds.append(g - h)
        ds = np.array(ds)
        out[name] = float(np.degrees(np.arctan2(np.sin(ds).mean(), np.cos(ds).mean())))
    out['balance_point_deg'] = 0.5 * (out['left'] + out['right'])
    return out


KERNELS = [('EPG', 'EPG', ('EPG',), ('EPG',)),
           ('EPG', 'Delta7', ('EPG',), ('Delta7',)),
           ('Delta7', 'EPG', ('Delta7',), ('EPG',)),
           ('PEN', 'EPG', ('PEN',), ('EPG',)),
           ('EPG', 'PFL3', ('EPG',), ('PFL3',))]


def kernel_set(net, W, types):
    ph = net.phase
    out = {}
    for a, b, pa, pb in KERNELS:
        pre = select(types, *pa, exact=True) if a != 'PEN' else select(types, 'PEN')
        post = select(types, *pb, exact=True)
        x, y = phase_kernel(W, ph, pre, post)
        out[f'{a}->{b}'] = y.tolist()
    out['bins_deg'] = x.tolist()
    return out


def fc2_pfl3_kernel(net, W, types, heading_ref):
    """FC2->PFL3 per side, x = FC2 phase minus the PFL3 cell's heading-input phase."""
    ph = net.phase
    fc2 = select(types, 'FC2')
    fc2 = fc2[ph[fc2] > -8]
    bins = 16
    out = {}
    for side, name in ((-1, 'left'), (+1, 'right')):
        pfl3 = [c for c in select(types, 'PFL3', exact=True)
                if net.side[c] == side and c in heading_ref]
        acc = np.zeros(bins)
        cnt = np.zeros(bins)
        for c in pfl3:
            d = wrap(ph[fc2] - heading_ref[c])
            k = np.floor((d + np.pi) / TAU * bins).astype(int) % bins
            np.add.at(acc, k, W[c, fc2])
            np.add.at(cnt, k, 1)
        out[name] = (acc / np.maximum(cnt, 1)).tolist()
    return out


def heading_reference(D):
    W = dense(D)
    ph = D['phase']
    epg = select(D['types'], 'EPG', exact=True)
    epg = epg[ph[epg] > -8]
    ref = {}
    for c in select(D['types'], 'PFL3', exact=True):
        w = np.maximum(W[c, epg], 0)
        if w.sum() > 0:
            ref[c] = np.arctan2((w * np.sin(ph[epg])).sum(), (w * np.cos(ph[epg])).sum())
    return ref


def alive_mask(net, keep):
    types = [net.type_of(i) for i in range(net.N)]
    if keep is None:
        return np.ones(net.N, dtype=bool)
    return np.array([any(t.startswith(p) for p in keep) for t in types])


def budget(net, W, alive):
    Wc = W.tocoo()
    ok = alive[Wc.row] & alive[Wc.col] & (Wc.data != 0)
    return int(alive.sum()), int(ok.sum())


def settle_turn(net, goal, seed=1, steps=300):
    net.reset(seed)
    for _ in range(steps):
        net.clear_drive()
        net.set_visual_scene(0.0, 1.0)
        net.inject_goal(goal, GOAL_GAIN)
        net.step(DT)
    return net.read_steering()['turn']


def closed_loop_errors(net, seeds=8, goals=6, steps=700):
    """Python replication of closedLoopErrors() in scripts/audit_claims.mjs."""
    errs = []
    for s in range(seeds):
        for k in range(goals):
            mag = np.radians(30 + 140 * (k + 0.5) / goals)
            goal = (1 if k % 2 else -1) * mag
            net.reset(1 + s)
            th = 0.0
            for _ in range(150):
                net.clear_drive(); net.set_visual_scene(th, 1.0)
                net.inject_goal(goal, GOAL_GAIN); net.step(DT)
            for _ in range(steps):
                net.clear_drive(); net.set_visual_scene(th, 1.0)
                net.inject_goal(goal, GOAL_GAIN); net.step(DT)
                th = float(wrap(th + K_TURN * net.read_steering()['turn'] * DT))
            errs.append(float(np.degrees(wrap(th - goal))))
    return errs


def boot_median_ci(v, n=2000, seed=12345):
    rng = np.random.default_rng(seed)
    v = np.asarray(v)
    m = [np.median(v[rng.integers(0, len(v), len(v))]) for _ in range(n)]
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def goal_trace(net):
    """Goal heading switched three times; the network produces all steering."""
    schedule = [(0.0, 0.0), (2.0, 90.0), (5.0, -60.0), (8.0, 150.0)]
    T = 11.0
    net.reset(1)
    th = 0.0
    rows = []
    for i in range(int(T / DT)):
        t = i * DT
        goal = np.radians([g for (t0, g) in schedule if t >= t0][-1])
        net.clear_drive(); net.set_visual_scene(th, 1.0)
        net.inject_goal(goal, GOAL_GAIN); net.step(DT)
        st = net.read_steering()
        th = float(wrap(th + K_TURN * st['turn'] * DT))
        rows.append([t, np.degrees(goal), np.degrees(th),
                     np.degrees(net.read_heading()['theta']), st['left'], st['right'], st['turn']])
    return {'schedule': schedule, 'rows': np.array(rows).tolist()}


def measure(quick=False):
    D = load_npz()
    net = CXNetworkNP(MODEL)
    types = [net.type_of(i) for i in range(net.N)]
    res = {'note': 'Measured by paper/scripts/fig_a.py from model/flybrain-cx.gguf and '
                   'data/cx_network.npz. Weights are never changed on disk.'}

    ring = ring_embedding(D)
    order, mean = glomerulus_order(ring)
    res['ring'] = {'n_epg': int(len(ring['idx'])), 'eigenvalues': ring['eig'],
                   'max_dev_from_stored_deg': ring['max_dev_deg'],
                   'glomerulus_order': order,
                   'glomerulus_angle_deg': {g: float(np.degrees(v)) for g, v in mean.items()}}

    res['pfl3_offsets_deg'] = pfl3_offsets(D)

    W8 = net.W
    Wd = {b: requantize(W8, b).toarray() for b in (8, 4, 3, 2)}
    href = heading_reference(D)
    res['kernels'] = {str(b): kernel_set(net, Wd[b], types) for b in (8, 4, 3, 2)}
    res['fc2_pfl3_kernel'] = {str(b): fc2_pfl3_kernel(net, Wd[b], types, href)
                              for b in (8, 4, 3, 2)}
    res['synapses_by_bits'] = {str(b): int((Wd[b] != 0).sum()) for b in (8, 4, 3, 2)}
    s8 = res['synapses_by_bits']['8']
    res['pruned_fraction_by_bits'] = {k: 1 - v / s8 for k, v in res['synapses_by_bits'].items()}

    # the minimal-circuit table: budget and steering-sign correctness
    configs = [('Intact', None, 8),
               ('Core (EPG + Delta7 + PFL3 + FC2)', ['EPG', 'Delta7', 'PFL3', 'FC2'], 4),
               ('EPG + FC2 + PFL3', ['EPG', 'PFL3', 'FC2'], 4),
               ('FC2 restricted to FC2A', ['EPG', 'PFL3', 'FC2A'], 4),
               ('No FC2', ['EPG', 'PFL3'], 8),
               ('Intact at 2 bit', None, 2)]
    table = []
    for name, keep, bits in configs:
        Wq = requantize(W8, bits)
        net.W = Wq
        net.invalidate_weight_split()
        alive = alive_mask(net, keep)
        net.alive = alive
        n, e = budget(net, Wq, alive)
        ok = tot = 0
        for s in range(1, 13):
            for g in (0.9, -0.9, 1.6, -1.6):
                t = settle_turn(net, g, seed=s)
                tot += 1
                ok += int(np.sign(t) == np.sign(g) and abs(t) > 0.05)
        errs = np.abs(closed_loop_errors(net)) if not quick else np.array([np.nan])
        med = float(np.median(errs))
        table.append({'config': name, 'neurons': n, 'synapses': e, 'bits': bits,
                      'kbit': e * bits / 1000, 'steering_sign_correct': ok, 'trials': tot,
                      'closed_loop_median_abs_deg': med,
                      'closed_loop_ci95': boot_median_ci(errs) if not quick else None,
                      'pass': bool(med < PASS_DEG)})
        print(f'  {name:36s} {n:5d} neurons {e:7d} syn {bits} bit  sign ok {ok}/{tot}'
              f'  closed loop {med:.1f} deg')
    res['minimal_circuit_table'] = table

    # steering as a function of goal offset (intact 8 bit vs minimal 4 bit)
    offs = np.arange(-180, 181, 10)
    curves = {}
    for name, keep, bits in (('intact', None, 8), ('minimal', ['EPG', 'PFL3', 'FC2'], 4)):
        net.W = requantize(W8, bits); net.invalidate_weight_split()
        net.alive = alive_mask(net, keep)
        curves[name] = [settle_turn(net, np.radians(o)) for o in offs]
    res['turn_vs_goal'] = {'offset_deg': offs.tolist(), **curves}

    net.W = W8; net.invalidate_weight_split(); net.alive = np.ones(net.N, dtype=bool)
    res['trace'] = goal_trace(net)

    if not quick:
        rep = {}
        for bits in (8, 4, 3, 2):
            net.W = requantize(W8, bits); net.invalidate_weight_split()
            e = closed_loop_errors(net)
            a = np.abs(e)
            rep[str(bits)] = {'median_abs_deg': float(np.median(a)), 'ci95': boot_median_ci(a),
                              'median_signed_deg': float(np.median(e)), 'n': len(e)}
            print(f'  python replication {bits} bit: {rep[str(bits)]["median_abs_deg"]:.1f} deg')
        res['closed_loop_python'] = rep
        net.W = W8; net.invalidate_weight_split()

    save_json(res, 'paper_a_measurements.json')
    return res


# ---------------------------------------------------------------- figures
def box(ax, x, y, w, h, title, body, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.08',
                                fc='white', ec=ec, lw=1.8))
    ax.text(x + w / 2, y + h - 0.22, title, ha='center', va='top', fontsize=10.5,
            weight='bold', color=ec)
    ax.text(x + w / 2, y + h - 0.62, body, ha='center', va='top', fontsize=8.4,
            color=INK, linespacing=1.5)


def arrow(ax, x0, y0, x1, y1, color=INK, rad=0.0, ls='-'):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=1.6, ls=ls,
                                connectionstyle=f'arc3,rad={rad}'))


def figA0():
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 4.6); ax.axis('off')
    box(ax, 0.2, 1.0, 2.6, 2.9, '1  Wiring (data)',
        'MaleCNS v1.0 connectome\n2,308 CX neurons\n187,137 signed edges\n\n'
        'w = synapse count\n x sign(transmitter)\nACh +1, GABA/Glu -1', BLUE)
    box(ax, 3.2, 1.0, 2.6, 2.9, '2  Rate model',
        '(assumed, not in the data)\none rate r in [0,1] per cell\n\ntau dr/dt = -r + f(syn)\n'
        'syn = g Sum w r + b + I\n\nf saturating, threshold,\ntime constants: imposed', ORANGE)
    box(ax, 6.2, 1.0, 2.6, 2.9, '3  Declared I/O',
        'clamp: ER, ExR, LNO, SpsP\n(afferent lines)\n\ninject: EPG <- visual cue\n'
        '         FC2 <- goal\n\nread: EPG bump -> heading\n'
        '        PFL3 R-L -> turn', PURPLE)
    box(ax, 9.2, 1.0, 2.6, 2.9, '4  Closed loop',
        'PFL3 turn command\n-> rotate the simulated fly\n-> the visual scene moves\n'
        '-> new EPG input\n\nwithout the loop it is\nonly an ODE solution', GREEN)
    for x in (2.8, 5.8, 8.8):
        arrow(ax, x + 0.02, 2.45, x + 0.38, 2.45)
    arrow(ax, 10.5, 1.0, 7.5, 1.0, color=GREEN, rad=-0.35)
    ax.text(9.0, 0.28, 'the world closes the loop', ha='center', fontsize=8.5, color=GREEN)
    fig.suptitle('What it means to "run" a wiring diagram: only box 1 comes from the connectome',
                 x=0.012, ha='left', fontsize=13, weight='bold')
    save(fig, P, 'figA0_model')


def figA1(res, ring):
    fig = plt.figure(figsize=(12, 6.4))
    ax = fig.add_subplot(1, 2, 1, projection='polar')
    th = ring['theta']
    col = [ORANGE if g.startswith('L') else BLUE for g in ring['glom']]
    rr = 1 + 0.05 * np.sin(7 * np.arange(len(th)))
    ax.scatter(th, rr, c=col, s=60, edgecolor='white', lw=0.6, zorder=3)
    for g, a in res['ring']['glomerulus_angle_deg'].items():
        ax.text(np.radians(a), 1.24 if g.startswith('L') else 1.42, g, ha='center', va='center', fontsize=9, weight='bold',
                color=ORANGE if g.startswith('L') else BLUE)
    ax.set_ylim(0, 1.5); ax.set_yticks([])
    ax.set_title('A  EPG cells placed by recovered angle', loc='left', fontsize=11,
                 weight='bold', pad=18)
    ax.scatter([], [], c=ORANGE, label='left PB glomerulus'); ax.scatter([], [], c=BLUE,
                                                                       label='right PB glomerulus')
    ax.legend(loc='lower left', bbox_to_anchor=(-0.08, -0.12), frameon=False, fontsize=8.5)

    bx = fig.add_subplot(1, 2, 2)
    bx.scatter(ring['v1'], ring['v2'], c=col, s=40, edgecolor='white', lw=0.5)
    bx.set_aspect('equal')
    bx.axhline(0, color=GRID, lw=1); bx.axvline(0, color=GRID, lw=1)
    bx.set_xlabel('eigenvector 2  (cos mode)'); bx.set_ylabel('eigenvector 3  (sin mode)')
    panel_title(bx, 'B  Leading non-uniform eigenvectors form a circle')
    ev = res['ring']['eigenvalues']
    note(bx, f'Row-normalized two-hop EPG->EPG graph (unsigned).\nEigenvalues: '
             f'{ev[0]:.2f} (uniform), {ev[1]:.3f}, {ev[2]:.3f}, {ev[3]:.3f} ...\n'
             f'angle = atan2(v3, v2); no anatomical coordinate is used.\nMax difference from '
             f'the phase stored in the model: {res["ring"]["max_dev_from_stored_deg"]:.1f} deg.', y=-0.13)
    order = res['ring']['glomerulus_order']
    fig.suptitle('Heading phase recovered from connectivity alone: left and right glomeruli interleave',
                 x=0.012, ha='left', fontsize=13, weight='bold')
    fig.text(0.012, 0.905, 'Glomeruli in order of recovered angle: ' + ' '.join(order),
             fontsize=9, color=DIM)
    fig.subplots_adjust(left=0.05, right=0.97, top=0.78, bottom=0.26, wspace=0.3)
    save(fig, P, 'figA1_ring')


def figA2(res):
    k = res['kernels']['8']
    x = np.array(k['bins_deg'])
    fk = res['fc2_pfl3_kernel']['8']
    off = res['pfl3_offsets_deg']
    fig, axs = plt.subplots(2, 3, figsize=(13, 7.2))
    notes = {'EPG->EPG': 'same-heading excitation (local)',
             'EPG->Delta7': 'to the opposite side (180 deg)',
             'Delta7->EPG': 'inhibition: global inhibition loop',
             'PEN->EPG': 'shifted by about +/-45-67 deg: rotates the bump',
             'EPG->PFL3': 'current heading, unshifted (0 deg)'}
    for ax, key in zip(axs.ravel()[:5], notes):
        y = np.array(k[key])
        ax.bar(x, y, width=20, color=[RED if v < 0 else BLUE for v in y])
        ax.axhline(0, color=GRID, lw=1)
        ax.set_xticks([-180, -90, 0, 90, 180]); ax.set_xlim(-190, 190)
        panel_title(ax, key.replace('->', ' -> ').replace('Delta7', 'Delta7'))
        ax.text(0.02, 0.97, notes[key], transform=ax.transAxes, fontsize=8.3, color=DIM, va='top')
        ax.set_xlabel('post - pre phase (deg)', fontsize=8.5)
        ax.set_ylabel('mean normalized weight', fontsize=8.5)
    ax = axs[1, 2]
    ax.plot(x, fk['left'], '-o', color=ORANGE, ms=4, label='left PFL3')
    ax.plot(x, fk['right'], '-o', color=PURPLE, ms=4, label='right PFL3')
    ax.axvline(off['left'], color=ORANGE, ls=':', lw=1.3)
    ax.axvline(off['right'], color=PURPLE, ls=':', lw=1.3)
    ax.axhline(0, color=GRID, lw=1)
    ax.set_xticks([-180, -90, 0, 90, 180]); ax.set_xlim(-190, 190)
    panel_title(ax, 'FC2 -> PFL3, split by side')
    ax.legend(frameon=False, fontsize=8.5, loc='upper right')
    ax.text(0.02, 0.97, f'left {off["left"]:+.1f} deg / right {off["right"]:+.1f} deg\n'
                        'this offset is the steering mechanism',
            transform=ax.transAxes, fontsize=8.3, color=DIM, va='top')
    ax.set_xlabel('FC2 phase - PFL3 heading-input phase (deg)', fontsize=8.5)
    fig.suptitle('Connection kernels: 187,137 edges binned by phase difference '
                 '(MaleCNS v1.0, Q8_0 weights)', x=0.012, ha='left', fontsize=13, weight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, P, 'figA2_kernels')


def figA3(res):
    fig = plt.figure(figsize=(13, 5.2))
    ax = fig.add_axes([0.02, 0.08, 0.44, 0.78]); ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis('off')
    panel_title(ax, 'A  PFL3 as a shifted comparator')
    ax.add_patch(FancyBboxPatch((0.3, 4.6), 2.8, 1.4, boxstyle='round,pad=0.05', fc='white', ec=BLUE, lw=2))
    ax.text(1.7, 5.3, 'EPG\ncurrent heading H', ha='center', va='center', fontsize=9.5, color=BLUE, weight='bold')
    ax.add_patch(FancyBboxPatch((0.3, 1.0), 2.8, 1.4, boxstyle='round,pad=0.05', fc='white', ec=ORANGE, lw=2))
    ax.text(1.7, 1.7, 'FC2\ngoal heading G', ha='center', va='center', fontsize=9.5, color=ORANGE, weight='bold')
    for y, name, sh in ((5.0, 'left PFL3', '-73'), (1.9, 'right PFL3', '+73')):
        ax.add_patch(FancyBboxPatch((6.0, y - 0.6), 3.4, 1.3, boxstyle='round,pad=0.05',
                                    fc='white', ec=PURPLE, lw=2))
        ax.text(7.7, y + 0.05, f'{name}\nH vs (G shifted {sh} deg)', ha='center', va='center',
                fontsize=9, color=PURPLE, weight='bold')
    arrow(ax, 3.2, 5.3, 5.95, 5.1, BLUE); arrow(ax, 3.2, 5.1, 5.95, 2.2, BLUE)
    arrow(ax, 3.2, 1.9, 5.95, 4.8, ORANGE); arrow(ax, 3.2, 1.7, 5.95, 1.8, ORANGE)
    ax.text(4.6, 6.2, 'unshifted (0 deg)', fontsize=8.5, color=BLUE, ha='center')
    ax.text(4.6, 0.6, 'shifted -73 / +73 deg', fontsize=8.5, color=ORANGE, ha='center')
    ax.text(5.0, 3.45, 'turn = (R - L) / (R + L)', fontsize=10, ha='center', color=INK,
            bbox=dict(fc='#f4f5f7', ec='none', pad=4))
    ax.text(0.3, -0.2, 'Coincidence detection: the side whose shifted goal overlaps the heading bump wins.\n'
                       'Subtraction G - H is done by where the inputs sit in phase, not by a subtractor.',
            fontsize=8.3, color=DIM)

    bx = fig.add_axes([0.55, 0.22, 0.42, 0.6])
    tv = res['turn_vs_goal']
    x = np.array(tv['offset_deg'])
    bx.plot(x, tv['intact'], '-o', color=INK, ms=3.5, label='intact, 8 bit (2,308 neurons)')
    bx.plot(x, tv['minimal'], '--o', color=ORANGE, ms=3.5, label='EPG + FC2 + PFL3, 4 bit')
    bx.plot(x, np.sin(np.radians(x)), color=GRID, lw=5, zorder=0, label='sin(G - H) for reference')
    bx.axhline(0, color='#9aa3ad', lw=0.8); bx.axvline(0, color='#9aa3ad', lw=0.8)
    bx.set_xticks([-180, -90, 0, 90, 180]); bx.set_xlim(-185, 185)
    bx.set_xlabel('goal offset G - H (deg)'); bx.set_ylabel('steering command (R - L)/(R + L)')
    panel_title(bx, 'B  Open-loop steering response')
    bx.legend(frameon=False, fontsize=8.3, loc='upper left')
    note(bx, 'Visual cue fixed at 0 deg; goal injected at the model\'s default sites; 300 steps (6 s) '
             'to settle.\nPositive = counter-clockwise. The intact circuit falls silent at +90 deg '
             '(both PFL3 below threshold)\nand its sign flips beyond about +/-170 deg.', y=-0.14)
    fig.suptitle('The steering circuit: current heading vs goal, compared with a 73 deg phase shift',
                 x=0.012, ha='left', fontsize=13, weight='bold')
    save(fig, P, 'figA3_comparator')


def figA4(res):
    r = np.array(res['trace']['rows'])
    t = r[:, 0]
    fig, axs = plt.subplots(3, 1, figsize=(11, 7.4), sharex=True,
                            gridspec_kw={'height_ratios': [2, 1.2, 1.2]})
    ax = axs[0]
    ax.plot(t, r[:, 1], color=ORANGE, lw=2.4, label='goal heading (set from outside)')
    ax.plot(t, r[:, 2], color=BLUE, lw=2.2, label="fly's actual heading")
    ax.plot(t, r[:, 3], color=INK, lw=1.1, ls='--', label='EPG bump (heading inside the model)')
    ax.set_ylim(-190, 190); ax.set_yticks([-180, -90, 0, 90, 180]); ax.set_ylabel('heading (deg)')
    ax.legend(frameon=False, fontsize=8.5, loc='lower left', ncol=3)
    ax = axs[1]
    ax.plot(t, r[:, 4], color=ORANGE, lw=1.8, label='left PFL3')
    ax.plot(t, r[:, 5], color=PURPLE, lw=1.8, label='right PFL3')
    ax.set_ylabel('summed rate'); ax.legend(frameon=False, fontsize=8.5, loc='upper right')
    ax = axs[2]
    ax.plot(t, r[:, 6], color=INK, lw=1.3)
    ax.fill_between(t, 0, r[:, 6], where=r[:, 6] > 0, color=PURPLE, alpha=0.25, lw=0)
    ax.fill_between(t, 0, r[:, 6], where=r[:, 6] < 0, color=ORANGE, alpha=0.25, lw=0)
    ax.axhline(0, color=GRID)
    ax.set_ylabel('steering\n(R - L)/(R + L)'); ax.set_xlabel('time (s)')
    ax.text(0.01, 0.9, 'purple = counter-clockwise, orange = clockwise', transform=ax.transAxes,
            fontsize=8, color=DIM, va='top')
    for a in axs:
        for t0, _ in res['trace']['schedule'][1:]:
            a.axvline(t0, color='#c9ced4', ls=':', lw=1)
    fig.suptitle('Closed loop: the goal changes three times and the model produces all of the steering',
                 x=0.012, ha='left', fontsize=13, weight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, P, 'figA4_trace')


def figA5():
    g = GGUFReader(MODEL)
    size = {26: 4, 27: 8, 0: 4}
    rows = []
    for name, info in g.tensors.items():
        n = int(np.prod(info['dims']))
        b = n // 32 * 34 if info['type'] == 8 else n * size[info['type']]
        kind = {26: 'I32', 27: 'I64', 0: 'F32', 8: 'Q8_0'}[info['type']]
        rows.append((name, kind, n, b))
    total = sum(r[3] for r in rows)
    fig = plt.figure(figsize=(13, 5.4))
    ax = fig.add_axes([0.1, 0.16, 0.45, 0.68])
    names = [r[0] for r in rows][::-1]
    vals = [r[3] / 1024 for r in rows][::-1]
    cols = [ORANGE if r[1] == 'Q8_0' else (PURPLE if r[0] == 'cx.phase' else BLUE) for r in rows][::-1]
    ax.barh(names, vals, color=cols)
    for i, (v, r) in enumerate(zip(vals, rows[::-1])):
        ax.text(v + 3, i, f'{r[1]}  {r[2]:,} values', va='center', fontsize=8, color=DIM)
    ax.set_xlabel('size (KiB)'); ax.set_xlim(0, max(vals) * 1.45)
    fsize = os.path.getsize(MODEL)
    panel_title(ax, f'A  Tensors in flybrain-cx.gguf (file {fsize:,} bytes)')
    note(ax, 'orange = Q8_0 weights (quantized), purple = cx.phase (F32, never quantized),\n'
             'blue = sparse-matrix indices and per-cell parameters', y=-0.13)

    bx = fig.add_axes([0.62, 0.14, 0.36, 0.7]); bx.set_xlim(0, 10); bx.set_ylim(0, 7); bx.axis('off')
    panel_title(bx, 'B  Q8_0: one block of 32 weights')
    for i in range(32):
        bx.add_patch(plt.Rectangle((0.2 + i * 0.29, 5.0), 0.26, 0.8, fc=BLUE, alpha=0.25 + 0.5 * ((i * 7) % 10) / 10))
    bx.text(5.0, 6.2, '32 x float32 = 128 bytes', ha='center', fontsize=9.5)
    arrow(bx, 5.0, 4.8, 5.0, 3.7)
    bx.text(5.25, 4.2, 'd = max|w| / 127;  q = round(w / d)', fontsize=9)
    bx.add_patch(plt.Rectangle((0.2, 2.4), 0.9, 0.8, fc=PURPLE))
    bx.text(0.65, 2.8, 'f16\nd', ha='center', va='center', fontsize=8, color='white')
    for i in range(32):
        bx.add_patch(plt.Rectangle((1.2 + i * 0.26, 2.4), 0.23, 0.8, fc=ORANGE, alpha=0.8))
    bx.text(5.0, 1.8, '2 + 32 x int8 = 34 bytes  (3.8x smaller)', ha='center', fontsize=9.5)
    bx.text(0.2, 0.6, 'The graph is stored as CSR (indptr, indices, weights).\n'
                      f'Source tables 1.05 GB  ->  {fsize / 1e6:.2f} MB .gguf (about 1/1000).', fontsize=8.6, color=DIM)
    fig.suptitle('Packing the connectome into GGUF: weights quantized, coordinates exact',
                 x=0.012, ha='left', fontsize=13, weight='bold')
    save(fig, P, 'figA5_gguf')
    return {'tensors': [{'name': r[0], 'type': r[1], 'n': r[2], 'bytes': r[3]} for r in rows],
            'tensor_bytes': total, 'file_bytes': fsize}


def figA6(res):
    audit = load_json('claims_audit_maze.json')['claims']['bit_depth_heading_error_deg']
    x = np.array(res['kernels']['8']['bins_deg'])
    cols = {'8': INK, '4': ORANGE, '3': BLUE, '2': RED}
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.6))
    ax = axs[0]
    for b in ('8', '4', '3', '2'):
        ax.plot(x, res['kernels'][b]['EPG->EPG'], '-o', ms=3, color=cols[b], label=f'{b} bit')
    ax.set_xticks([-180, -90, 0, 90, 180]); ax.set_xlabel('post - pre phase (deg)')
    ax.set_ylabel('mean normalized weight'); ax.legend(frameon=False)
    panel_title(ax, 'A  EPG -> EPG (the bump-forming peak)')
    ax = axs[1]
    for b in ('8', '4', '3', '2'):
        k = res['fc2_pfl3_kernel'][b]
        ax.plot(x, k['left'], '-', color=cols[b], lw=1.6, label=f'{b} bit')
        ax.plot(x, k['right'], '--', color=cols[b], lw=1.6)
    off = res['pfl3_offsets_deg']
    for o in (off['left'], off['right']):
        ax.axvline(o, color='#9aa3ad', ls=':')
    ax.set_xticks([-180, -90, 0, 90, 180]); ax.set_xlabel('FC2 phase - heading-input phase (deg)')
    panel_title(ax, 'B  FC2 -> PFL3 (solid = left, dashed = right)')
    ax.legend(frameon=False, fontsize=8.5)
    ax = axs[2]
    bits = ['8', '4', '3', '2']
    med = [audit[b]['abs']['median'] for b in bits]
    lo = [audit[b]['abs']['median'] - audit[b]['abs']['ci95'][0] for b in bits]
    hi = [audit[b]['abs']['ci95'][1] - audit[b]['abs']['median'] for b in bits]
    xs = np.arange(4)
    ax.errorbar(xs, med, yerr=[lo, hi], fmt='o-', color=INK, capsize=4, ms=7, lw=2)
    for i, m in enumerate(med):
        ax.text(i + 0.1, m + 5, f'{m:.1f}', fontsize=9)
    ax.axhline(90, color=RED, ls='--', lw=1); ax.text(0.05, 92, 'chance (90 deg)', color=RED, fontsize=8.5)
    ax.set_xticks(xs); ax.set_xticklabels([f'{b} bit' for b in bits])
    ax.set_ylabel('closed-loop |heading error| (deg)'); ax.set_ylim(0, 120)
    panel_title(ax, 'C  Performance cliff between 3 and 2 bit')
    note(ax, 'median of 48 trials (8 seeds x 6 goals), bootstrap 95% CI.\n'
             'data/claims_audit_maze.json (JavaScript engine).', y=-0.16)
    syn = res['synapses_by_bits']
    fig.suptitle('Requantizing in memory: the curve keeps its shape down to 3 bit '
                 f'(non-zero synapses: 8 bit {syn["8"]:,}, 4 bit {syn["4"]:,}, 3 bit {syn["3"]:,}, '
                 f'2 bit {syn["2"]:,})', x=0.012, ha='left', fontsize=12, weight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, P, 'figA6_bits')


def figA7(res):
    tab = res['minimal_circuit_table']
    fig, ax = plt.subplots(figsize=(12, 4.6))
    names = [r['config'] for r in tab][::-1]
    kb = [r['kbit'] for r in tab][::-1]
    cols = [GREEN if r['pass'] else RED for r in tab][::-1]
    ax.barh(names, kb, color=cols)
    ax.set_xscale('log'); ax.set_xlabel('kbit = surviving synapses x bits (log scale)')
    for i, r in enumerate(tab[::-1]):
        ax.text(r['kbit'] * 1.15, i,
                f"{r['neurons']:,} neurons / {r['synapses']:,} syn / {r['bits']} bit   "
                f"closed-loop error {r['closed_loop_median_abs_deg']:.1f} deg   "
                f"sign probe {r['steering_sign_correct']}/{r['trials']}",
                va='center', fontsize=8.2, color=INK)
    ax.set_xlim(1, 3e6)
    panel_title(ax, 'How small can the maze-competent brain get?  (green = pass, red = fail)')
    note(ax, f'Pass = median closed-loop |heading error| < {PASS_DEG:.0f} deg over 48 trials '
             '(8 seeds x 6 goals). Sign probe = open-loop steering sign correct\n'
             '(12 seeds x goals +/-0.9, +/-1.6 rad, |turn| > 0.05), as in scripts/audit_claims.mjs.',
         y=-0.16)
    fig.tight_layout()
    save(fig, P, 'figA7_minimal')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    res = measure(args.quick)
    D = load_npz()
    ring = ring_embedding(D)
    figA0()
    figA1(res, ring)
    figA2(res)
    figA3(res)
    figA4(res)
    res['gguf'] = figA5()
    figA6(res)
    figA7(res)
    save_json(res, 'paper_a_measurements.json')


if __name__ == '__main__':
    main()
