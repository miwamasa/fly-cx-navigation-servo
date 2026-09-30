#!/usr/bin/env python3
"""Figures and the number sheet for paper B (coordinate transformation and feedback control).

  python3 paper/scripts/fig_b.py        # seconds; reads data/*.json only

Outputs
  paper/data/paper_b_measurements.json     every number paper B cites, with its source file
  paper/b_servo_control/figures/figB*.png  English figures

Paper B uses only this project's own measurements and theory. The augmented
three-mechanism model discussed in docs/PAPER_SERVO.*.md was reported by another
system and never reproduced here, so none of its numbers appear in these figures.
"""

from __future__ import annotations

import json
import os

import numpy as np
from matplotlib.patches import FancyBboxPatch, Patch

from common import (BLUE, DIM, GREEN, GRID, INK, ORANGE, PAPER, PURPLE, RED, load_json,
                    note, panel_title, plt, save, save_json)

P = 'b_servo_control'
K_PLANT = 2.6
GROUP_COL = {'PFNd-L': BLUE, 'PFNd-R': '#7fb8d4', 'PFNv-L': ORANGE, 'PFNv-R': '#e0a36a'}
GROUPS = ['PFNd-L', 'PFNd-R', 'PFNv-L', 'PFNv-R']


def arrow(ax, x0, y0, x1, y1, color=INK, rad=0.0, lw=1.6, ls='-'):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, ls=ls,
                                connectionstyle=f'arc3,rad={rad}'))


def box(ax, x, y, w, h, text, ec=INK, fs=9.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.1',
                                fc='white', ec=ec, lw=1.8))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=ec,
            weight='bold')


# ---------------------------------------------------------------- number sheet
def collect():
    """Gather the numbers paper B cites into one file, each with its source."""
    st = load_json('servo_theory.json')
    si = load_json('servo_identify.json')
    ih = load_json('servo_ideal_hdb.json')
    bh = load_json('bearing_vs_homing.json')
    pi = load_json('pfl3_inputs.json')
    pb = load_json('pfn_basis.json')
    pm = load_json('pfn_model.json')
    sh = load_json('shunting.json')
    cl = load_json('coord_localize.json')
    ct = load_json('coord_transform.json')
    mech = load_json('pi_mechanism.json')
    conf = load_json('servo_confound.json')
    net = next(iter(si['network'].values()))['fit']
    nb = list(bh['network'].values())
    g = st['G_task']
    ideal_rows = {(r['kind'], r['amplitude']): r['fit'] for r in ih['rows']}
    coord_b = [r['fit']['travel_gain'] for r in ct['results'] if r['fit']]
    sh_rows = {r['shunt_frac'] if r['shunt_gain'] == 1.0 and r['sign_variant'] == 'measured'
               else None: r for r in sh['rows']}
    f1 = sh_rows[1.0]
    loc = cl['conditions'][0]['stages']
    return {
        'source_note': 'Values copied from data/*.json by paper/scripts/fig_b.py. '
                       'No number from the unreproduced augmented model is included.',
        'plant': {'K_rad_s': K_PLANT, 'sideslip_sigma_deg': st['task']['sigma_deg'],
                  'sideslip_tau_s': st['task']['tau_s'], 'sideslip_max_deg': st['task']['max_deg'],
                  'seed': st['task']['seed'], 'source': 'data/servo_theory.json'},
        'back_stage_fit': {'K_H': net['K_H'], 'K_T': net['K_T'], 'c': net['c'], 'r2': net['r2'],
                           'rho': net['rho'], 'source': 'data/servo_identify.json'},
        'synthetic_controls': {k: {'K_H': v['fit']['K_H'], 'K_T': v['fit']['K_T'],
                                   'rho': v['fit']['rho'], 'verdict': v['verdict']}
                               for k, v in si['synthetic'].items()},
        'ideal_hdb_clamp_rho': {k: ideal_rows[(k, 0.2)]['rho'] for k in ('travel', 'heading', 'shuffled')},
        'ideal_hdb_clamp_r2': {k: ideal_rows[(k, 0.2)]['r2'] for k in ('travel', 'heading', 'shuffled')},
        'goal_injection_confound_deg': {r['goal_types'].__str__(): r['median_deg'] for r in conf['rows']},
        'task_error_deg': {
            'circuit_sideslip': nb[0]['median_deg'], 'circuit_sideslip_ci95': nb[0]['ci95'],
            'circuit_no_sideslip': nb[1]['median_deg'], 'circuit_no_sideslip_ci95': nb[1]['ci95'],
            'ideal_heading_sideslip': g['ideal_heading']['sideslip']['median_deg'],
            'ideal_travel_sideslip': g['ideal_travel']['sideslip']['median_deg'],
            'ideal_travel_sideslip_p90': g['ideal_travel']['sideslip']['p90_deg'],
            'plain_law_sideslip': g['plain']['sideslip']['median_deg'],
            'plain_law_no_sideslip': g['plain']['no_sideslip']['median_deg'],
            'median_abs_phi': g['median_abs_phi_deg'],
            'floor': g['budget']['plain']['floor_deg'],
            'rss_prediction': g['budget']['plain']['rss_prediction_sideslip_deg'],
            'prediction_within_ci': g['budget']['plain']['prediction_within_ci'],
            'source': 'data/bearing_vs_homing.json, data/servo_theory.json'},
        'travel_gain_sweep': g['travel_gain_sweep'],
        'static_rows_plain': [r for r in st['A_static']['rows'] if r['model'] == 'plain'],
        'phasor': st['C_phasor'], 'comparator': st['D_comparator'],
        'pll': st['E_pll']['rows'], 'world_wind': st['F_worldwind']['rows'],
        'pfl3_routes': {k: v['share'] for k, v in pi['pfl3_routes'].items()},
        'hdb_to_pfl3': pi['direct']['hdb_to_pfl3'],
        'hdb_two_hop': pi['hdb_to_pfl3_two_hop'],
        'pfn_write_offset_deg': pm['measured']['write_offset_deg'],
        'pfn_concentration': pm['measured']['concentration'],
        'pfn_collapsed_pref_deg': pm['measured']['collapsed_pref_deg'],
        'n_hdb_all_four': pb['n_all_four'], 'n_hdb': pb['n_hdb_cells'],
        'pfn_pairwise_deg': pb['pairwise_deg'],
        'phasor_exact_err_deg': pm['exactness']['max_angle_err_deg'],
        'coord_transform_max_travel_gain': max(coord_b),
        'coord_transform_n_conditions': len(ct['results']),
        'coord_transform_n_pass': sum(1 for r in ct['results'] if r['verdict'].get('pass')),
        'shunting_rows': [{'f': r['shunt_frac'], 'k': r['shunt_gain'], 'sign': r['sign_variant'],
                           'maze_deg': r['maze_deg'],
                           'PFNd_alive_moving': r['probe']['PFNd']['alive_moving'],
                           'PFNv_alive_moving': r['probe']['PFNv']['alive_moving']}
                          for r in sh['rows']],
        'shunting_f1_maze_deg': f1['maze_deg'],
        'localize_f1': loc,
        'hdb_input_share_f1': cl['conditions'][0]['hdb_input_share'],
        'sideslip_uptake_k': {k: v['PFN'] for k, v in mech['conditions'].items()},
    }


# ---------------------------------------------------------------- figures
def figB1():
    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.set_xlim(0, 13); ax.set_ylim(0, 6.2); ax.axis('off')

    def loop(y0, fb, col, label, sub):
        ax.text(0.2, y0 + 1.55, label, fontsize=11.5, weight='bold', color=col)
        ax.text(0.2, y0 + 1.2, sub, fontsize=8.8, color=DIM)
        ax.add_patch(plt.Circle((1.6, y0 + 0.4), 0.22, fc='white', ec=INK, lw=1.5))
        ax.text(1.6, y0 + 0.4, '-', ha='center', va='center', fontsize=13, weight='bold')
        ax.text(0.45, y0 + 0.5, 'goal G', fontsize=10, color=INK)
        arrow(ax, 1.05, y0 + 0.4, 1.36, y0 + 0.4)
        box(ax, 2.3, y0, 2.4, 0.8, f'comparator\nu = K sin(G - {fb})', ec=PURPLE, fs=9)
        arrow(ax, 1.84, y0 + 0.4, 2.28, y0 + 0.4)
        box(ax, 5.4, y0, 2.0, 0.8, 'body turn\ndH/dt = K u', ec=INK, fs=9)
        arrow(ax, 4.72, y0 + 0.4, 5.38, y0 + 0.4)
        ax.add_patch(plt.Circle((8.4, y0 + 0.4), 0.22, fc='white', ec=INK, lw=1.5))
        ax.text(8.4, y0 + 0.4, '+', ha='center', va='center', fontsize=12, weight='bold')
        arrow(ax, 7.42, y0 + 0.4, 8.16, y0 + 0.4)
        ax.text(7.55, y0 + 0.55, 'H', fontsize=10, color=BLUE, weight='bold')
        ax.text(8.25, y0 + 1.05, 'sideslip phi', fontsize=9.5, color=RED)
        arrow(ax, 8.4, y0 + 1.0, 8.4, y0 + 0.64, color=RED)
        arrow(ax, 8.64, y0 + 0.4, 9.6, y0 + 0.4)
        ax.text(9.7, y0 + 0.33, 'travel T = H + phi', fontsize=10, color=ORANGE, weight='bold')
        if fb == 'H':
            ax.plot([7.8, 7.8, 1.6, 1.6], [y0 + 0.4, y0 - 0.45, y0 - 0.45, y0 + 0.18], color=BLUE, lw=1.8)
            ax.text(3.2, y0 - 0.4, 'feeds back H: phi is outside the loop', fontsize=9, color=BLUE, va='bottom')
        else:
            ax.plot([9.3, 9.3, 1.6, 1.6], [y0 + 0.4, y0 - 0.45, y0 - 0.45, y0 + 0.18], color=ORANGE, lw=1.8)
            ax.text(3.2, y0 - 0.4, 'feeds back T: phi is inside the loop', fontsize=9, color=ORANGE, va='bottom')

    loop(4.0, 'H', BLUE, 'Heading servo   u = K sin(G - H)',
         'equilibrium H* = G, so T* = G + phi: the whole sideslip stays as steady error')
    loop(1.1, 'T', ORANGE, 'Travel servo   u = K sin(G - T)',
         'with constant phi, dT/dt = dH/dt, so de/dt = -K sin e (e = G - T): error goes to 0')
    ax.text(0.2, 0.05, 'Read as a first-order phase-locked loop: phase detector = comparator, VCO (integrator 1/s) = '
                       'body turn, input phase = G, own phase = T,\nfrequency offset = dphi/dt.',
            fontsize=8.8, color=DIM)
    fig.suptitle('Where the disturbance enters: a crosswind (sideslip) is added at the plant output',
                 x=0.012, ha='left', fontsize=13, weight='bold')
    save(fig, P, 'figB1_loop')


def phase_line(e, phi, KH, KT):
    return -K_PLANT * (KH * np.sin(e + phi) + KT * np.sin(e))


def figB2(nums):
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.8))
    e = np.linspace(-np.pi, np.pi, 721)
    phi = np.radians(40)
    fit = nums['back_stage_fit']
    ax = axs[0]
    cases = [('ideal heading servo (K_H=1, K_T=0)', 1.0, 0.0, BLUE, '-'),
             ('ideal travel servo (K_H=0, K_T=1)', 0.0, 1.0, ORANGE, '-'),
             (f"circuit fit (K_H={fit['K_H']:+.3f}, K_T={fit['K_T']:+.3f})", fit['K_H'], fit['K_T'], INK, '--'),
             ('hypothetical K_H=0.2, K_T=-0.4 (sum < 0)', 0.2, -0.4, PURPLE, ':')]
    for lab, kh, kt, c, ls in cases:
        ax.plot(np.degrees(e), phase_line(e, phi, kh, kt), color=c, ls=ls, lw=2, label=lab)
    ax.axhline(0, color='#9aa3ad', lw=0.8)
    ax.set_xticks([-180, -90, 0, 90, 180]); ax.set_xlabel('e = G - T (deg)'); ax.set_ylabel('de/dt (rad/s)')
    panel_title(ax, 'A  Phase line at phi = 40 deg')
    ax.legend(frameon=False, fontsize=7.4, loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=1)
    ax.set_ylim(-3, 3)

    ax = axs[1]
    rows = nums['static_rows_plain']
    ph = [r['phi_deg'] for r in rows]
    ax.plot([0, 60], [0, 60], color=BLUE, lw=5, alpha=0.3, label='ideal heading servo: |e*| = phi')
    ax.plot(ph, [r['exact_deg'] for r in rows], 'o-', color=INK, label='circuit fit, exact equilibrium')
    ax.plot(ph, [r['linear_deg'] for r in rows], 's--', color=DIM, ms=4, label='circuit fit, K_H/(K_H+K_T) phi')
    ax.plot([0, 60], [0, 0], color=ORANGE, lw=2, label='ideal travel servo: 0')
    ax.set_xlabel('constant sideslip phi (deg)'); ax.set_ylabel('steady error |G - T| (deg)')
    panel_title(ax, 'B  Steady error = (1 - rho) phi')
    ax.legend(frameon=False, fontsize=7.8, loc='upper left')
    note(ax, f"rho = |K_T|/(|K_H|+|K_T|) = {fit['rho']:.3f} for the circuit:\nit rejects about "
             f"{100 * fit['rho']:.0f}% of the sideslip. The small excess\nover phi comes from the negative K_T.", y=-0.16)

    ax = axs[2]
    sw = nums['travel_gain_sweep']
    K = [r['K_loop'] for r in sw]
    ax.plot(K, [r['median_deg'] for r in sw], 'o-', color=ORANGE, label='ideal travel servo, simulated')
    ax.plot(K, [r['linear_median_deg'] for r in sw], '--', color=DIM, label='linear OU formula (no clip)')
    ax.axvline(K_PLANT, color='#9aa3ad', ls=':')
    ax.text(K_PLANT * 1.05, 17, 'K = 2.6 rad/s\n(the task)', fontsize=8, color=DIM)
    ax.set_xscale('log'); ax.set_xlabel('loop gain K (rad/s)'); ax.set_ylabel('median |G - T| (deg)')
    panel_title(ax, 'C  Fluctuating sideslip leaves a tracking lag')
    ax.legend(frameon=False, fontsize=7.8)
    note(ax, 'OU sideslip (sigma 55 deg, tau 4 s, clipped at 60 deg).\nEven a perfect travel servo lags a moving '
             'disturbance;\nthe lag shrinks as the loop gain rises (Appendix A.1).', y=-0.16)
    fig.suptitle('The mixed control law u = K_H sin(G - H) + K_T sin(G - T): what rho means and '
                 'what it cannot remove', x=0.012, ha='left', fontsize=13, weight='bold')
    fig.subplots_adjust(left=0.05, right=0.98, top=0.86, bottom=0.36, wspace=0.3)
    save(fig, P, 'figB2_theory')


def theory_panels():
    """Figures 25 and 26 of docs/ are already English and free of external numbers: reuse them."""
    import fig_servo_theory as fst
    plt.rcParams['font.family'] = 'DejaVu Sans'   # the module asks for a CJK font first
    d = load_json('servo_theory.json')
    out = os.path.join(PAPER, P, 'figures')
    fst.LANG = 'en'
    fst.OUTDIR = out
    fst.fig25(d)
    fst.fig26(d)
    os.replace(os.path.join(out, 'fig25_ops_en.png'), os.path.join(out, 'figB3_operations.png'))
    os.replace(os.path.join(out, 'fig26_pll_en.png'), os.path.join(out, 'figB4_pll.png'))
    print('-> figB3_operations.png, figB4_pll.png (from scripts/fig_servo_theory.py)')


def figB5(nums):
    op = load_json('pfn_operating.json')['scan']['measured']
    ct = load_json('coord_transform.json')
    fig = plt.figure(figsize=(17, 5.8))
    ax = fig.add_axes([0.03, 0.14, 0.25, 0.6], projection='polar')
    lit = {'PFNd-L': 45, 'PFNd-R': -45, 'PFNv-L': -135, 'PFNv-R': 135}
    for gname in GROUPS:
        w = np.radians(nums['pfn_write_offset_deg'][gname])
        R = nums['pfn_concentration'][gname]
        ax.annotate('', xy=(w, 1.0), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='-|>', color=GROUP_COL[gname], lw=2.4))
        ax.plot([np.radians(lit[gname])] * 2, [0, 1.0], color=GROUP_COL[gname], ls=':', lw=1.2)
        ax.text(w, 1.2, f"{gname}\n{nums['pfn_write_offset_deg'][gname]:+.0f} deg (R {R:.2f})",
                ha='center', va='center', fontsize=8, color=GROUP_COL[gname], weight='bold')
    ax.set_ylim(0, 1.4); ax.set_yticks([])
    ax.set_title('A  Four write offsets onto hDeltaB\n(solid = measured, dotted = literature)',
                 loc='left', fontsize=10.5, weight='bold', pad=22)

    bx = fig.add_axes([0.35, 0.2, 0.25, 0.6])
    gains = [r['pfn_gain'] for r in op]
    for gname, col in (('PFNd', BLUE), ('PFNv', ORANGE)):
        bx.plot(gains, [r['groups'][gname]['alive_moving'] for r in op], '-o', color=col, ms=4,
                label=f'{gname}: fraction active while walking')
        bx.plot(gains, [r['groups'][gname]['heading_follow_still'] for r in op], '--s', color=col, ms=4,
                label=f'{gname}: heading tuning at rest')
    bx.set_xscale('log'); bx.set_xlabel('PFN row gain'); bx.set_ylim(-0.03, 1.08)
    panel_title(bx, 'B  No operating point has both')
    bx.legend(frameon=True, framealpha=0.9, edgecolor='none', fontsize=7.6, loc='center right')
    note(bx, 'All self-motion entries (LNO, SpsP) are inhibitory, and so is the Delta7 input that\n'
             'carves heading tuning: gain that survives one flattens the other. data/pfn_operating.json',
         y=-0.17)

    cx = fig.add_axes([0.8, 0.2, 0.19, 0.6])
    names_en = {
        'measured': 'as measured', 'selfmotion': 'self-motion excitatory', 'glut': 'glutamate excitatory'}
    rows = [r for r in ct['results'] if r['fit']]
    y = np.arange(len(rows))
    b = [r['fit']['travel_gain'] for r in rows]
    a = [r['fit']['heading_gain'] for r in rows]
    cx.axvspan(0.8, 1.2, color=GREEN, alpha=0.12)
    cx.scatter(b, y, color=ORANGE, s=36, zorder=3, label='travel gain b')
    cx.scatter(a, y, color=BLUE, s=24, marker='s', zorder=3, label='heading gain a')
    labels = []
    for r in rows:
        p = r['params']
        s = names_en.get(p['sign_variant'], p['sign_variant'])
        if p.get('shunt_frac'):
            s += f", shunt f={p['shunt_frac']}"
        if p.get('shunt_gain') not in (None, 1.0):
            s += f" k={p['shunt_gain']}"
        if p.get('hd_gain', 1.0) != 1.0:
            s += f", hD gain {p['hd_gain']:g}"
        if p.get('pref_on') == 'pfn':
            s += ', pref on PFN'
        if 'pfnv_gain' in p:
            s += f", PFNv gain {p['pfnv_gain']}"
        labels.append(s)
    cx.set_yticks(y); cx.set_yticklabels(labels, fontsize=7)
    cx.axvline(0, color='#9aa3ad', lw=0.8)
    cx.set_xlabel('fitted gain in psi = a theta + b phi + c')
    panel_title(cx, 'C  hDeltaB tracks heading, not travel')
    cx.legend(frameon=False, fontsize=7.6, loc='lower right')
    note(cx, f"green band = pass (a, b in [0.8, 1.2]).\n{nums['coord_transform_n_pass']} of "
             f"{nums['coord_transform_n_conditions']} conditions pass; max b = "
             f"{nums['coord_transform_max_travel_gain']:+.2f}.\nSilent-readout conditions omitted. "
             "data/coord_transform.json", y=-0.12)
    fig.suptitle('Front stage (PFNd/PFNv -> hDeltaB): the wiring for the transform is present, '
                 'the operation is not', x=0.012, ha='left', fontsize=13, weight='bold')
    save(fig, P, 'figB5_front_stage')


def figB6(nums):
    rows = [r for r in nums['shunting_rows'] if r['k'] == 1.0 and r['sign'] == 'measured']
    fig = plt.figure(figsize=(16, 5.6))
    ax = fig.add_subplot(1, 3, 1)
    f = [r['f'] for r in rows]
    ax.plot(f, [r['PFNd_alive_moving'] for r in rows], '-o', color=BLUE, label='PFNd active while walking')
    ax.plot(f, [r['PFNv_alive_moving'] for r in rows], '-o', color=ORANGE, label='PFNv active while walking')
    ax.set_xlabel('shunting fraction f (0 = subtractive, 1 = divisive)'); ax.set_ylabel('fraction of cells')
    ax2 = ax.twinx()
    ax2.plot(f, [r['maze_deg'] for r in rows], '--s', color=RED, label='closed-loop maze error (deg)')
    ax2.set_ylabel('maze error (deg)', color=RED); ax2.spines['right'].set_visible(True)
    ax2.set_ylim(0, 110)
    panel_title(ax, 'A  Shunting wakes PFNv, and breaks the maze')
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7.6, loc='center left')

    bx = fig.add_subplot(1, 3, 2, projection='polar')
    lit = {'PFNd-L': 45, 'PFNd-R': -45, 'PFNv-L': -135, 'PFNv-R': 135}
    for gname in GROUPS:
        a0 = np.radians(lit[gname])
        a1 = np.radians(nums['pfn_collapsed_pref_deg'][gname])
        bx.plot([a0, a0], [0, 1], color=GROUP_COL[gname], ls=':', lw=1.4)
        bx.annotate('', xy=(a1, 1.0), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='-|>', color=GROUP_COL[gname], lw=2.4))
        rr = {'PFNd-L': 1.25, 'PFNd-R': 1.2, 'PFNv-L': 1.3, 'PFNv-R': 1.62}[gname]
        bx.text(a1, rr, f"{gname} {nums['pfn_collapsed_pref_deg'][gname]:+.0f}", ha='center',
                fontsize=8, color=GROUP_COL[gname], weight='bold')
    bx.set_ylim(0, 1.75); bx.set_yticks([])
    bx.set_title('B  Preferences collapse forward (f = 1)\n(dotted = given to the entry, arrow = measured)',
                 loc='left', fontsize=10.5, weight='bold', pad=22)

    cx = fig.add_subplot(1, 3, 3)
    loc = nums['localize_f1']
    st = [('four-basis sum\n(measured amp. x offsets)', loc['basis']), ('PFN population\nvector', loc['pfn']),
          ('hDeltaB population\nvector', loc['hdb'])]
    x = np.arange(3)
    cx.bar(x - 0.18, [s[1]['heading_gain'] for s in st], 0.36, color=BLUE, label='heading gain a')
    cx.bar(x + 0.18, [s[1]['travel_gain'] for s in st], 0.36, color=ORANGE, label='travel gain b')
    cx.axhspan(0.8, 1.2, color=GREEN, alpha=0.12)
    cx.set_xticks(x); cx.set_xticklabels([s[0] for s in st], fontsize=8)
    cx.axhline(0, color='#9aa3ad', lw=0.8)
    panel_title(cx, 'C  The signal dies at the first stage')
    cx.legend(frameon=False, fontsize=7.8, loc='upper right')
    share = nums['hdb_input_share_f1']
    note(cx, f"PFN supplies {100 * (share['PFNd'] + share['PFNv']):.0f}% of hDeltaB's excitatory input\n"
             f"(PFNd {100 * share['PFNd']:.0f}%, PFNv {100 * share['PFNv']:.0f}%). "
             'Nothing is diluted on the way;\nPFN itself carries heading only.', y=-0.2)
    fig.suptitle('Declaring shunting (divisive) inhibition: the stage comes into service, '
                 'the transformation still does not', x=0.012, ha='left', fontsize=13, weight='bold')
    fig.subplots_adjust(left=0.05, right=0.98, top=0.74, bottom=0.24, wspace=0.45)
    save(fig, P, 'figB6_shunting')


def figB7(nums):
    fig, axs = plt.subplots(1, 3, figsize=(17, 6.0))
    ax = axs[0]
    names = {'hD_other': 'hDelta (not hDeltaB)', 'G': 'goal: FC1/FC2/FS/FR', 'H': 'heading: EPG/Delta7/PEN',
             'PFN': 'PFN', 'other': 'other', 'vD': 'vDelta', 'T': 'travel: hDeltaB (direct)'}
    routes = sorted(nums['pfl3_routes'].items(), key=lambda kv: kv[1])
    cols = {'T': RED, 'H': BLUE, 'G': ORANGE}
    ax.barh([names[k] for k, _ in routes], [100 * v for _, v in routes],
            color=[cols.get(k, '#b7bec7') for k, _ in routes])
    for i, (k, v) in enumerate(routes):
        ax.text(100 * v + 0.5, i, f'{100 * v:.2f}%' if v < 0.01 else f'{100 * v:.1f}%', va='center', fontsize=8)
    ax.set_xlabel('share of PFL3 input (|synapses|)')
    panel_title(ax, 'A  What PFL3 receives')
    d = nums['hdb_to_pfl3']
    note(ax, f"hDeltaB -> PFL3: {d['n_synapses']} synapses in {d['n_connections']} connections. "
             'Wiring, not activity.\ndata/pfl3_inputs.json', y=-0.17)

    bx = axs[1]
    syn = nums['synthetic_controls']
    fit = nums['back_stage_fit']
    ideal = nums['ideal_hdb_clamp_rho']
    lab = ['synthetic\nheading', 'synthetic\nmixed', 'synthetic\ntravel', 'circuit', 'clamp ideal\ntravel on hDB',
           'clamp heading\non hDB', 'clamp shuffled\non hDB']
    rho = [syn['heading']['rho'], syn['mixed']['rho'], syn['travel']['rho'], fit['rho'],
           ideal['travel'], ideal['heading'], ideal['shuffled']]
    col = [DIM, DIM, DIM, INK, ORANGE, BLUE, PURPLE]
    lab = [x.replace('\n', ' ') for x in lab][::-1]
    rho = rho[::-1]; col = col[::-1]
    bx.barh(range(7), rho, color=col)
    for i, r in enumerate(rho):
        bx.text(r + 0.02, i, f'{r:.3f}', va='center', fontsize=8)
    bx.set_yticks(range(7)); bx.set_yticklabels(lab, fontsize=8)
    bx.axvline(0.2, color=BLUE, ls=':', lw=1); bx.axvline(0.8, color=ORANGE, ls=':', lw=1)
    bx.text(0.21, 6.45, 'heading < 0.2', fontsize=7.5, color=BLUE)
    bx.text(0.81, 6.45, 'travel > 0.8', fontsize=7.5, color=ORANGE)
    bx.set_xlabel('rho = |K_T| / (|K_H| + |K_T|)'); bx.set_xlim(0, 1.15)
    panel_title(bx, 'B  The closed loop is a heading servo')
    note(bx, f"circuit fit: K_H = {fit['K_H']:+.3f}, K_T = {fit['K_T']:+.3f}, R^2 = {fit['r2']:.3f}.\n"
             'Synthetic controls recover their known answer;\nan ideal travel signal clamped onto hDeltaB '
             'changes nothing.', y=-0.14)

    cx = axs[2]
    t = nums['task_error_deg']
    items = [('ideal\nheading\nservo', t['ideal_heading_sideslip'], None, BLUE),
             ('ideal\ntravel\nservo', t['ideal_travel_sideslip'], None, ORANGE),
             ('circuit\nfit\n(law only)', t['plain_law_sideslip'], None, DIM),
             ('RSS\nprediction\nlaw + floor', t['rss_prediction'], None, PURPLE),
             ('circuit\n(measured)', t['circuit_sideslip'], t['circuit_sideslip_ci95'], INK)]
    for i, (lab, v, ci, c) in enumerate(items):
        cx.bar(i, v, color=c, alpha=0.9)
        if ci:
            cx.errorbar(i, v, yerr=[[v - ci[0]], [ci[1] - v]], color='white', capsize=4, lw=1.5)
        cx.text(i, v + 1, f'{v:.1f}', ha='center', fontsize=8.5)
    cx.set_xticks(range(5)); cx.set_xticklabels([x[0] for x in items], fontsize=7.8)
    cx.set_ylabel('median |G - T| with sideslip (deg)')
    panel_title(cx, 'C  Theory predicts the measured error')
    note(cx, f"floor = sqrt({t['circuit_no_sideslip']:.1f}^2 - {t['plain_law_no_sideslip']:.1f}^2) = "
             f"{t['floor']:.1f} deg:\nthe error without sideslip that the law does not explain.\n"
             f"Prediction {t['rss_prediction']:.1f} deg lies inside the measured CI "
             f"[{t['circuit_sideslip_ci95'][0]:.1f}, {t['circuit_sideslip_ci95'][1]:.1f}].", y=-0.24)
    fig.suptitle('Back stage (-> PFL3) and the closed loop: the comparator is fed heading, not travel',
                 x=0.012, ha='left', fontsize=13, weight='bold')
    fig.subplots_adjust(left=0.13, right=0.98, top=0.85, bottom=0.3, wspace=0.6)
    save(fig, P, 'figB7_back_stage')


def figB8(nums):
    k = nums['sideslip_uptake_k']
    keys = list(k)
    labels = ['baseline\n(as measured)', '(1) equalize\nL/R drive', '(2) swap\nL/R drive', '(3) shuffle PFN\nphase labels']
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    v = [k[x]['k'] for x in keys]
    ci = [k[x]['ci95'] for x in keys]
    cols = [INK, BLUE, ORANGE, PURPLE]
    ax.bar(range(4), v, yerr=ci, color=cols, capsize=5, alpha=0.9)
    for i, (a, c) in enumerate(zip(v, ci)):
        ax.text(i, max(a, 0) + c + 0.012, f'{a:+.3f} +/- {c:.3f}', fontsize=8.5, ha='center')
    ax.axhline(0, color='#9aa3ad', lw=0.8); ax.set_ylim(-0.2, 0.22)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel('uptake k in  (population angle - theta) = k beta + c')
    panel_title(ax, 'Sideslip reaches PFN only as a left/right drive ratio')
    note(ax, '20 trajectories x 20 independent sets, 95% CI. k = 0: heading only; k = 1: travel direction '
             'only.\ndata/pi_mechanism.json', y=-0.2)
    fig.tight_layout()
    save(fig, P, 'figB8_uptake')


# ---------------------------------------------------------------- section 7: the added circuit
VARIANT_COL = {'N': PURPLE, 'C': GREEN}
VARIANT_NAME = {'N': 'N (decoded angles)', 'C': 'C (circuit, no decoding)'}


def collect_augmented():
    """Numbers of section 7, copied from data/servo_augmented.json (None if not yet run)."""
    path = os.environ.get('SERVO_AUGMENTED_JSON',
                          os.path.join(os.path.dirname(PAPER), 'data', 'servo_augmented.json'))
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    out = {'source': 'data/servo_augmented.json', 'quick': d.get('quick', False),
           'prereg': d['prereg'], 'ideal': d['ideal'],
           'plain': {'back': d['plain']['back'],
                     'sideslip': {k: d['plain']['sideslip'][k] for k in ('median_deg', 'ci95', 'p90_deg')},
                     'no_sideslip': {k: d['plain']['no_sideslip'][k] for k in ('median_deg', 'ci95', 'p90_deg')}},
           'plain_robustness': {k: {'median_deg': v['median_deg'], 'ci95': v['ci95']}
                                for k, v in d['plain_robustness'].items()},
           'variants': {}}
    for v, rec in d['variants'].items():
        e = rec['evaluation']
        cfg = rec['config']
        out['variants'][v] = {
            'config': {k: cfg[k] for k in ('m', 'nu_P', 'nu_B', 'lam', 'c_T_deg')},
            'M_nonzero': rec['M_nonzero'], 'M_pfn_cells': rec['M_pfn_cells'],
            'M_hdb_cells': rec['M_hdb_cells'],
            'front_train': {**e['front_train']['fit'], 'amp_r2': e['front_train']['amplitude']['r2'],
                            'pass': e['front_train']['verdict']['pass']},
            'front_held_out': {**e['front_held_out']['fit'],
                               'amp_r2': e['front_held_out']['amplitude']['r2'],
                               'pass': e['front_held_out']['verdict']['pass']},
            'back': {**e['back']['fit'], 'pass': e['back']['verdict']['pass']},
            'sideslip': {k: e['closed_loop']['sideslip'][k] for k in ('median_deg', 'ci95', 'p90_deg')},
            'no_sideslip': {k: e['closed_loop']['no_sideslip'][k] for k in ('median_deg', 'ci95', 'p90_deg')},
            'trials_improved': int(sum(a < b for a, b in zip(
                e['closed_loop']['sideslip']['per_trial_median_deg'],
                d['plain']['sideslip']['per_trial_median_deg']))),
            'n_trials': len(e['closed_loop']['sideslip']['per_trial_median_deg']),
            'closed_loop_pass': e['closed_loop']['pass'],
            'budget': e['budget'],
            'ablations': {k: {'travel_gain': (r.get('front') or {}).get('travel_gain'),
                              'K_H': r['back']['K_H'], 'K_T': r['back']['K_T'],
                              'rho': r['back']['rho'], 'r2': r['back']['r2'],
                              'sideslip_deg': r['sideslip']['median_deg'],
                              'ci95': r['sideslip']['ci95']}
                          for k, r in rec['ablations'].items()},
            'robustness': {k: {'median_deg': r['median_deg'], 'ci95': r['ci95']}
                           for k, r in rec['robustness'].items()},
        }
    return out


def figB9():
    fig, ax = plt.subplots(figsize=(13.5, 6.4))
    ax.set_xlim(0, 13.5); ax.set_ylim(0, 6.4); ax.axis('off')
    box(ax, 0.3, 3.2, 2.2, 1.0, 'self-motion\n|v|, phi', ec=INK, fs=9)
    box(ax, 0.3, 4.8, 2.2, 1.0, 'EPG\nheading H', ec=BLUE, fs=9)
    box(ax, 3.4, 4.0, 2.6, 1.3, '(1) PFN\nA_g x heading gate', ec=PURPLE, fs=9)
    box(ax, 6.9, 4.0, 2.6, 1.3, '(2) hDeltaB\n+ q M r_PFN, M >= 0', ec=PURPLE, fs=9)
    box(ax, 10.4, 2.4, 2.8, 1.3, '(3) PFL3\nlam [g + t]^2', ec=PURPLE, fs=9)
    box(ax, 6.9, 1.2, 2.6, 1.0, 'FC2  goal G', ec=ORANGE, fs=9)
    arrow(ax, 2.5, 3.7, 3.4, 4.4); arrow(ax, 2.5, 5.3, 3.4, 4.9, BLUE)
    arrow(ax, 6.0, 4.65, 6.9, 4.65)
    arrow(ax, 9.5, 4.3, 10.6, 3.7, ORANGE, ls='--')
    arrow(ax, 9.5, 1.7, 10.5, 2.5, ORANGE)
    ax.text(9.9, 4.25, 'T', fontsize=11, color=ORANGE, weight='bold')
    ax.text(12.6, 1.9, 'turn u = (R-L)/(R+L)', fontsize=9, ha='center')
    ax.text(0.3, 0.55,
            'Variant N (decoded): gate = 1 + cos(H^ - alpha_j) with H^ from the EPG population vector; '
            'g = cos(G^ - gamma_i), t = cos(T^ - gamma_i - delta)\n'
            'with G^, T^ decoded from FC2 and hDeltaB.   '
            'Variant C (circuit): gate = each PFN cell\'s own EPG synaptic input; g = each PFL3 cell\'s own '
            'FC2 input;\nt = a newly declared hDeltaB -> PFL3 cosine projection (dashed; the connectome has '
            'only 32 synapses here).   (2) is identical in both: M >= 0 on existing pairs only.',
            fontsize=8.6, color=DIM, va='bottom')
    fig.suptitle('Section 7: the two missing multiplications, added to the connectome model',
                 x=0.012, ha='left', fontsize=13, weight='bold')
    save(fig, P, 'figB9_added_circuit')


def figB10(aug):
    V = list(aug['variants'])
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.4))
    ax = axs[0]
    labels = []
    x = 0
    for v in V:
        for split in ('front_train', 'front_held_out'):
            f = aug['variants'][v][split]
            ax.bar(x - 0.2, f['heading_gain'], 0.18, color=BLUE)
            ax.bar(x, f['travel_gain'], 0.18, color=ORANGE)
            ax.bar(x + 0.2, f['amp_r2'], 0.18, color=DIM)
            labels.append(f'{v} {"train" if split == "front_train" else "held-out"}')
            x += 1
    ax.axhspan(0.8, 1.2, color=GREEN, alpha=0.1)
    ax.axhline(0.9, color=DIM, ls=':', lw=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8)
    ax.legend(handles=[Patch(color=BLUE, label='heading gain a'), Patch(color=ORANGE, label='travel gain b'),
                       Patch(color=DIM, label='speed amplitude R^2 (pass >= 0.9, dotted)')],
              frameon=False, fontsize=7.6, loc='upper left', ncol=2)
    ax.set_ylim(0, 1.5)
    panel_title(ax, 'A  Front stage: hDeltaB now carries T')

    bx = axs[1]
    rows = [('plain circuit', aug['plain']['back'], INK)] + \
           [(VARIANT_NAME[v], aug['variants'][v]['back'], VARIANT_COL[v]) for v in V]
    for i, (lab, f, c) in enumerate(rows):
        bx.bar(i - 0.2, f['K_H'], 0.38, color=BLUE)
        bx.bar(i + 0.2, f['K_T'], 0.38, color=ORANGE)
        bx.text(i, max(f['K_H'], f['K_T']) + 0.04, f"rho {f['rho']:.3f}", ha='center', fontsize=8.5)
    bx.axhline(0, color='#9aa3ad', lw=0.8)
    bx.set_xticks(range(len(rows))); bx.set_xticklabels([r[0] for r in rows], fontsize=8)
    bx.legend(handles=[Patch(color=BLUE, label='K_H'), Patch(color=ORANGE, label='K_T')],
              frameon=False, fontsize=8, loc='upper left')
    bx.set_ylim(min(0, min(min(r[1]['K_H'], r[1]['K_T']) for r in rows)) - 0.05,
                max(max(r[1]['K_H'], r[1]['K_T']) for r in rows) + 0.2)
    panel_title(bx, 'B  Back stage: from heading to travel servo')

    cx = axs[2]
    items = [('ideal heading', aug['ideal']['heading'], None, BLUE),
             ('plain circuit', aug['plain']['sideslip']['median_deg'], aug['plain']['sideslip']['ci95'], INK)]
    items += [(v, aug['variants'][v]['sideslip']['median_deg'], aug['variants'][v]['sideslip']['ci95'],
               VARIANT_COL[v]) for v in V]
    items += [('ideal travel', aug['ideal']['travel'], None, ORANGE)]
    for i, (lab, val, ci, c) in enumerate(items):
        cx.bar(i, val, color=c, alpha=0.9)
        if ci:
            cx.errorbar(i, val, yerr=[[val - ci[0]], [ci[1] - val]], color='white', capsize=4, lw=1.5)
        cx.text(i, val + 1, f'{val:.1f}', ha='center', fontsize=8.5)
    no = [aug['plain']['no_sideslip']['median_deg']] + [aug['variants'][v]['no_sideslip']['median_deg'] for v in V]
    cx.scatter(range(1, 2 + len(V)), no, marker='_', s=500, color=RED, zorder=4, label='without sideslip')
    cx.axhline(20, color=GREEN, ls=':', lw=1); cx.text(len(items) - 0.5, 21, 'pass <= 20', fontsize=8, color=GREEN, ha='right')
    cx.set_xticks(range(len(items))); cx.set_xticklabels([x[0] for x in items], fontsize=8)
    cx.set_ylabel('median |G - T| with sideslip (deg)')
    cx.legend(frameon=False, fontsize=8, loc='upper right')
    panel_title(cx, 'C  Closed loop with sideslip')
    fig.suptitle('Performance of the added circuit: front stage, back stage and closed loop',
                 x=0.012, ha='left', fontsize=13, weight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, P, 'figB10_augmented')


def figB11(aug):
    V = list(aug['variants'])
    fig, axs = plt.subplots(1, 2, figsize=(16, 7.2), gridspec_kw={'width_ratios': [1, 1.15]})
    for ax, key, title in ((axs[0], 'ablations', 'A  Ablations and controls'),
                           (axs[1], 'robustness', 'B  Robustness')):
        names = list(aug['variants'][V[0]][key])
        y = np.arange(len(names))
        for j, v in enumerate(V):
            rows = aug['variants'][v][key]
            med = [rows[n]['sideslip_deg'] if key == 'ablations' else rows[n]['median_deg'] for n in names]
            ci = [rows[n]['ci95'] for n in names]
            yy = y + (j - 0.5) * 0.28
            ax.errorbar(med, yy, xerr=[[m - c[0] for m, c in zip(med, ci)], [c[1] - m for m, c in zip(med, ci)]],
                        fmt='o', color=VARIANT_COL[v], ms=5, capsize=2, label=VARIANT_NAME[v])
        if key == 'robustness':
            base = aug['plain_robustness']
            ax.scatter([base[n]['median_deg'] if n in base else np.nan for n in names], y + 0.36,
                       marker='x', color=INK, s=22, label='plain circuit')
        ax.axvline(aug['variants'][V[0]]['sideslip']['median_deg'], color=PURPLE, ls=':', lw=0.8)
        ax.axvline(20, color=GREEN, ls=':', lw=1)
        ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8.2)
        ax.invert_yaxis()
        ax.set_xlabel('median |G - T| with sideslip (deg), 95% CI')
        ax.set_xscale('log'); ax.set_xticks([2, 5, 10, 20, 50, 100, 180])
        ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
        ax.legend(frameon=False, fontsize=8, loc='lower right')
        panel_title(ax, title)
    fig.suptitle('Which part is necessary, and how robust is the result?  (green line: pass <= 20 deg)',
                 x=0.012, ha='left', fontsize=13, weight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, P, 'figB11_ablation_robustness')


def main():
    nums = collect()
    aug = collect_augmented()
    if aug is not None:
        nums['augmented'] = aug
    save_json(nums, 'paper_b_measurements.json')
    figB1()
    figB2(nums)
    theory_panels()
    figB5(nums)
    figB6(nums)
    figB7(nums)
    figB8(nums)
    if aug is not None:
        figB9()
        figB10(aug)
        figB11(aug)


if __name__ == '__main__':
    main()
