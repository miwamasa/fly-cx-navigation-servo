#!/usr/bin/env python3
"""図 22〜26 — 制御理論による解析（docs/PAPER_SERVO.*.md）。日本語版と英語版を両方書く。

  fig22_loop      閉ループのブロック図。横滑りはどこから入り、三つの機構はどこに付くか。PLL との対応
  fig23_portrait  位相平面：素の回路 / 追加モデル / 符号が反転した線形比較
  fig24_budget    誤差の内訳：定常成分 (1−ρ)|φ|、ループ利得で決まる追従誤差、残りの床
  fig25_ops       前段と後段の演算：フェーザの掛け算（角度の足し算）と、二乗による比較（引き算）
  fig26_pll       PLL としての性質：ランプ追従誤差 arcsin((dφ/dt)/K)、ロック外れ、世界固定風

  python3 scripts/fig_servo_theory.py
数値は data/servo_theory.json と data/bearing_vs_homing.json から。

配色は dataviz の検証器で確認した 3 色（青 #2f88b0 / 橙 #c26a1a / 紫 #7a5fb5）に限る。
赤は「不安定」の状態色としてだけ使い、必ず白抜きの印と文字を添える（色だけで区別しない）。
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import servo_theory as st  # noqa: E402

plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
INK, DIM, GRID = '#1b2028', '#6d7681', '#e3e7ec'
HEAD, TRAV, THIRD = '#2f88b0', '#c26a1a', '#7a5fb5'
BAD = '#c94f4f'
OUTDIR = os.path.join(ROOT, 'docs', 'figures')

LANG = 'ja'


def T(ja, en):
    return ja if LANG == 'ja' else en


def load(n):
    return json.load(open(os.path.join(ROOT, 'data', n), encoding='utf-8'))


def title(ax, s):
    ax.set_title(s, loc='left', fontsize=11.5, weight='bold', color=INK)


def note(ax, s, y=-0.16):
    ax.text(0.0, y, s, transform=ax.transAxes, fontsize=8.0, color=DIM, va='top')


def clean(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(DIM)
    ax.tick_params(colors=DIM, labelsize=8.4)
    ax.grid(color=GRID, lw=.8)
    ax.set_axisbelow(True)


def box(ax, x, y, w, h, text, ec=INK, fc='white', fs=9, bold=True, color=None):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0.02',
                                fc=fc, ec=ec, lw=1.6))
    ax.text(x, y, text, ha='center', va='center', fontsize=fs,
            weight='bold' if bold else 'normal', color=color or ec)


def arrow(ax, x0, y0, x1, y1, color=INK, lw=1.6, ls='-', rad=0.0):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, ls=ls,
                                shrinkA=0, shrinkB=0,
                                connectionstyle=f'arc3,rad={rad}'))


def save(fig, name):
    fig.savefig(os.path.join(OUTDIR, f'{name}_{LANG}.png'), dpi=150)
    plt.close(fig)
    print(f'→ docs/figures/{name}_{LANG}.png')


# ======================================================================== 図 22

def fig22():
    fig = plt.figure(figsize=(13.8, 8.6))
    fig.patch.set_facecolor('white')
    ax = fig.add_axes([.02, .30, .96, .58])
    ax.set_xlim(0, 14); ax.set_ylim(0, 6); ax.axis('off')
    title(ax, T('A  閉ループ：横滑り φ はプラントの出口で足される',
                'A  Closed loop: sideslip φ is added at the plant output'))

    # 主経路
    box(ax, 1.0, 3.6, 1.2, .8, T('目標 G\n(FC2)', 'goal G\n(FC2)'), ec=INK)
    ax.add_patch(plt.Circle((2.6, 3.6), .28, fc='white', ec=INK, lw=1.6))
    ax.text(2.6, 3.6, '−', ha='center', va='center', fontsize=13, weight='bold')
    arrow(ax, 1.6, 3.6, 2.32, 3.6)
    box(ax, 4.6, 3.6, 2.6, 1.0, T('比較器  sin(G − ·)\n(PFL3 の左右差)',
                                   'comparator  sin(G − ·)\n(PFL3 L/R difference)'), ec=INK)
    arrow(ax, 2.88, 3.6, 3.3, 3.6)
    box(ax, 7.6, 3.6, 1.8, 1.0, T('体の回転\nḢ = K·u', 'body turn\nḢ = K·u'), ec=INK)
    ax.text(6.35, 3.85, 'u', fontsize=10, style='italic', color=INK)
    arrow(ax, 5.9, 3.6, 6.7, 3.6)
    ax.add_patch(plt.Circle((9.6, 3.6), .28, fc='white', ec=INK, lw=1.6))
    ax.text(9.6, 3.6, '+', ha='center', va='center', fontsize=13, weight='bold')
    arrow(ax, 8.5, 3.6, 9.32, 3.6)
    ax.text(8.85, 3.82, 'H', fontsize=10, style='italic', color=HEAD, weight='bold')
    arrow(ax, 9.88, 3.6, 12.6, 3.6)
    ax.text(12.7, 3.6, T('進行方向\nT = H + φ', 'travel\nT = H + φ'), va='center',
            fontsize=9.4, color=TRAV, weight='bold')
    # 外乱
    ax.text(9.6, 5.35, T('横滑り φ（外乱）', 'sideslip φ (disturbance)'), ha='center',
            fontsize=9.2, color=INK, weight='bold')
    arrow(ax, 9.6, 5.15, 9.6, 3.88)

    # heading 帰還（H を取る）
    ax.plot([8.9, 8.9], [3.6, 1.75], color=HEAD, lw=2.0, ls=(0, (5, 3)))
    ax.plot([8.9, 2.6], [1.75, 1.75], color=HEAD, lw=2.0, ls=(0, (5, 3)))
    arrow(ax, 2.6, 1.75, 2.6, 3.32, color=HEAD, lw=2.0, ls=(0, (5, 3)))
    ax.text(5.7, 1.45, T('heading servo：H を帰還（φ はループの外）→ 定常偏差 = φ',
                         'heading servo: feed back H (φ outside the loop) → steady error = φ'),
            ha='center', fontsize=9, color=HEAD, weight='bold')
    # travel 帰還（T を取る）
    ax.plot([11.6, 11.6], [3.6, .75], color=TRAV, lw=2.4)
    ax.plot([11.6, 2.45], [.75, .75], color=TRAV, lw=2.4)
    arrow(ax, 2.45, .75, 2.45, 3.33, color=TRAV, lw=2.4)
    ax.text(7.0, .42, T('travel servo：T を帰還（φ はループの内側）→ 定常偏差 → 0',
                        'travel servo: feed back T (φ inside the loop) → steady error → 0'),
            ha='center', fontsize=9, color=TRAV, weight='bold')

    # 三つの機構の付く場所
    tags = [(10.75, 2.55, '①', T('PFN の乗算で\nT を作る', 'PFN multiply\nbuilds T')),
            (10.75, 1.25, '②', T('位相補正で\nT の位相を合わせる', 'phase fix\naligns T')),
            (4.6, 4.75, '③', T('二乗で G − T を\n正しい符号で比べる', 'squaring compares\nG − T with sign'))]
    for x, y, n, s in tags:
        ax.add_patch(plt.Circle((x - .55, y), .2, fc=TRAV, ec='white', lw=1.2))
        ax.text(x - .55, y, n, ha='center', va='center', fontsize=9, color='white', weight='bold')
        ax.text(x - .25, y, s, va='center', fontsize=8.2, color=INK)
    ax.text(0.2, 5.55, T('素の配線では ① ② ③ が無く、比較器に入るのは H（EPG/Δ7 系 19.9%）。'
                         'hΔB → PFL3 はシナプス 32 個（0.09%）。',
                         'Plain wiring lacks ①②③; the comparator receives H (EPG/Δ7, 19.9%). '
                         'hΔB → PFL3 is 32 synapses (0.09%).'),
            fontsize=8.4, color=DIM)

    # PLL 対応
    bx = fig.add_axes([.04, .03, .92, .22]); bx.axis('off')
    title(bx, T('B  一次 PLL と同じ形：ė = −K·sin e',
                'B  Same form as a first-order PLL: ė = −K·sin e'))
    rows = [(T('PLL', 'PLL'), T('travel servo（ハエのモデル）', 'travel servo (fly model)')),
            (T('入力信号の位相', 'input phase'), T('目標方向 G', 'goal direction G')),
            (T('自分の信号の位相', 'own phase'), T('進行方向 T', 'travel direction T')),
            (T('位相比較器（掛け算）', 'phase detector (product)'),
             T('PFL3 の二乗比較 → sin(G − T)', 'PFL3 squaring → sin(G − T)')),
            (T('VCO（積分器 1/s）', 'VCO (integrator 1/s)'), T('体の回転 Ḣ = K·u', 'body turn Ḣ = K·u')),
            (T('周波数ずれ Δω', 'frequency offset Δω'), T('横滑りの変化率 dφ/dt', 'sideslip rate dφ/dt'))]
    for i, (a, b) in enumerate(rows):
        y = .80 - i * .155
        w = 'bold' if i == 0 else 'normal'
        bx.text(.02, y, a, fontsize=9, color=INK if i else DIM, weight=w, transform=bx.transAxes)
        bx.text(.30, y, b, fontsize=9, color=TRAV if i else DIM, weight=w, transform=bx.transAxes)
    bx.text(.62, .78, T('PLL で知られた性質がそのまま使える：',
                        'Known PLL properties carry over:'),
            fontsize=9, color=INK, weight='bold', transform=bx.transAxes)
    for i, s in enumerate([T('・位相差 π は不安定な平衡点', '• phase error π is an unstable equilibrium'),
                           T('・一定の dφ/dt には arcsin((dφ/dt)/K) の遅れが残る', '• constant dφ/dt leaves a lag arcsin((dφ/dt)/K)'),
                           T('・|dφ/dt| > K でロックが外れ、位相が周回する', '• |dφ/dt| > K breaks lock (cycle slips)'),
                           T('・一次ループなのでランプ入力に定常偏差が残る', '• type-1 loop: nonzero error to a ramp')]):
        bx.text(.62, .60 - i * .16, s, fontsize=8.8, color=INK, transform=bx.transAxes)

    fig.suptitle(T('横滑りを打ち消すには、制御したい量 T そのものを帰還する',
                   'To reject sideslip, feed back the controlled quantity T itself'),
                 fontsize=13.5, weight='bold', x=.012, ha='left', y=.975)
    save(fig, 'fig22_loop')


# ======================================================================== 図 23

def fig23(d):
    cases = [(T('素の回路（実測）', 'plain circuit (measured)'), 0.7483, -0.0185, HEAD),
             (T('追加モデル（GPT-6 報告）', 'augmented (GPT-6 report)'), 0.060, 0.703, TRAV),
             (T('線形比較（K_T < 0、代表値）', 'linear comparator (K_T < 0, representative)'),
              0.0155, -0.689, THIRD)]
    phi = np.radians(40)
    e = np.linspace(-np.pi, np.pi, 721)
    fig, axs = plt.subplots(1, 3, figsize=(13.8, 5.0), sharey=True)
    fig.patch.set_facecolor('white')
    for ax, (lab, KH, KT, col) in zip(axs, cases):
        f = -(KH * np.sin(e + phi) + KT * np.sin(e))
        ax.axhline(0, color=DIM, lw=1.0)
        ax.plot(np.degrees(e), f, color=col, lw=2.2)
        for r in st.equilibria(KH, KT, phi):
            x = np.degrees(r['e'])
            edge = abs(x) > 170
            xs = (-180, 180) if edge else (x,)
            lx, ha = ((-172, 'left') if edge else (x, 'center'))
            txt = '±180°' if edge else f'{x:+.0f}°'
            for xx in xs:
                if r['stable']:
                    ax.plot(xx, 0, 'o', ms=10, color=col, mec='white', mew=1.5, zorder=5, clip_on=False)
                else:
                    ax.plot(xx, 0, 'o', ms=10, mfc='white', mec=BAD, mew=2.0, zorder=5, clip_on=False)
            if r['stable']:
                ax.annotate(T(f'安定 {txt}', f'stable {txt}'), (xs[0], 0), (lx, .42),
                            ha=ha, fontsize=8.6, color=INK, weight='bold',
                            arrowprops=dict(arrowstyle='-', color=DIM, lw=.8))
            else:
                ax.annotate(T(f'不安定 {txt}', f'unstable {txt}'), (xs[-1], 0),
                            (x if not edge else 168, -.48 if not edge else .42),
                            ha='center' if not edge else 'right',
                            fontsize=8.6, color=BAD,
                            arrowprops=dict(arrowstyle='-', color=DIM, lw=.8))
        # 流れの向き
        for x0 in (-150, -90, -30, 30, 90, 150):
            v = -(KH * np.sin(np.radians(x0) + phi) + KT * np.sin(np.radians(x0)))
            if abs(v) > .05:
                ax.annotate('', xy=(x0 + 14 * np.sign(v), -0.02), xytext=(x0, -0.02),
                            arrowprops=dict(arrowstyle='-|>', color=DIM, lw=1.2))
        rho = st.rho(KH, KT)
        ax.text(0, 1.015, f'K_H = {KH:+.3f}   K_T = {KT:+.3f}   ρ = {rho:.3f}   '
                          f'K_H + K_T = {KH + KT:+.3f}', transform=ax.transAxes,
                fontsize=8.4, color=INK, va='bottom')
        ax.set_xlim(-180, 180); ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_ylim(-.85, .85)
        ax.set_xlabel(T('e = G − T（度）', 'e = G − T (deg)'), fontsize=9)
        clean(ax)
        ax.set_title(lab, loc='left', fontsize=10.5, weight='bold', color=col, pad=20)
    axs[0].set_ylabel(T('ė / K', 'ė / K'), fontsize=9)
    fig.suptitle(T('位相平面（φ = 40°）：ρ が大きくても K_H + K_T < 0 なら e ≈ 180° に落ちる',
                   'Phase line (φ = 40°): with K_H + K_T < 0 the loop settles near 180° even at high ρ'),
                 fontsize=13, weight='bold', x=.012, ha='left', y=.985)
    fig.text(.012, .915, T('ė = −K·[K_H·sin(e + φ) + K_T·sin e]。塗りの点が安定、白抜きが不安定。'
                           '右の係数は ρ を 0.978 に合わせた代表値で、GPT-6 の報告値の再現ではない。',
                           'ė = −K·[K_H·sin(e + φ) + K_T·sin e]. Filled = stable, hollow = unstable. '
                           'Right panel uses representative gains matched to ρ = 0.978, not the reported fit.'),
             fontsize=9, color=DIM)
    fig.subplots_adjust(left=.06, right=.985, top=.80, bottom=.12, wspace=.08)
    save(fig, 'fig23_portrait')


# ======================================================================== 図 24

def fig24(d):
    g = d['G_task']
    fig = plt.figure(figsize=(13.8, 5.4))
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(1, 3, wspace=.36, left=.06, right=.985, top=.80, bottom=.24)

    # A 定常偏差
    ax = fig.add_subplot(gs[0, 0])
    phis = np.linspace(0, 60, 61)
    for key, KH, KT, col, lab in (('plain', 0.7483, -0.0185, HEAD, T('素の回路', 'plain')),
                                  ('augmented', 0.060, 0.703, TRAV, T('追加モデル', 'augmented'))):
        ex = [st.stable_error(KH, KT, np.radians(p)) for p in phis]
        ax.plot(phis, ex, color=col, lw=2.2)
        ax.plot(phis, KH / (KH + KT) * phis, color=col, lw=1.2, ls=(0, (4, 3)))
        ax.text(61, ex[-1], lab, fontsize=8.8, color=INK, va='center', weight='bold')
    ax.set_xlim(0, 72); ax.set_ylim(0, 65)
    ax.set_xlabel(T('一定の横滑り φ（度）', 'constant sideslip φ (deg)'), fontsize=9)
    ax.set_ylabel(T('定常偏差 |G − T|（度）', 'steady error |G − T| (deg)'), fontsize=9)
    clean(ax)
    title(ax, T('A  ρ は外乱除去率', 'A  ρ is the rejection ratio'))
    note(ax, T('実線＝厳密解、破線＝線形近似 (1−ρ)φ。\nK_H, K_T > 0 のときだけ (1−ρ) に一致する。',
               'solid = exact, dashed = linear (1−ρ)φ.\nMatches (1−ρ) only when K_H, K_T > 0.'), y=-.19)

    # B 誤差の内訳
    ax = fig.add_subplot(gs[0, 1])
    bd = g['budget']
    rows = [
        (T('素の回路', 'plain'), [
            (T('制御則のみ', 'law only'), g['plain']['sideslip']['median_deg'], 'o'),
            (T('観測', 'observed'), bd['plain']['observed_sideslip_deg'], 's')], HEAD),
        (T('追加モデル', 'augmented'), [
            (T('定常成分 (1−ρ)|φ|', 'static (1−ρ)|φ|'), g['augmented']['static_component_median_deg'], 'D'),
            (T('制御則のみ', 'law only'), g['augmented']['sideslip']['median_deg'], 'o'),
            (T('観測（報告）', 'observed (reported)'), bd['augmented']['observed_sideslip_deg'], 's')], TRAV),
    ]
    y = 0
    yt, yl = [], []
    for grp, items, col in rows:
        for lab, v, mk in items:
            ax.plot([0, v], [y, y], color=GRID, lw=2.2, solid_capstyle='round')
            ax.plot(v, y, mk, ms=9, color=col, mec='white', mew=1.4, zorder=4)
            ax.text(v + 1.4, y, f'{v:.1f}°', va='center', fontsize=8.6, color=INK, weight='bold')
            yt.append(y); yl.append(f'{grp} · {lab}')
            y -= 1
        y -= .6
    ax.axvline(20, color=DIM, ls=':', lw=1.1)
    ax.text(20.5, .45, T('合格 ≤ 20°', 'pass ≤ 20°'), fontsize=7.8, color=DIM)
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=8.2)
    ax.set_xlim(0, 52); ax.set_ylim(y + .4, .8)
    ax.set_xlabel(T('横滑りありの誤差の中央値（度）', 'median error with sideslip (deg)'), fontsize=9)
    clean(ax); ax.grid(axis='y', visible=False)
    title(ax, T('B  15° のうち制御則で説明できるのは 9.5°', 'B  The law explains 9.5° of the 15°'))
    note(ax, T(f"残りを独立な床とみなすと（RSS）、追加モデルの床は {bd['augmented']['implied_floor_deg']:.1f}°。\n"
               f"素の回路で同じ仮定を使うと {bd['plain']['rss_prediction_sideslip_deg']:.1f}° を予測し、"
               f"観測 {bd['plain']['observed_sideslip_deg']:.1f}° の CI に入る。",
               f"Treating the rest as an independent floor (RSS): augmented floor "
               f"{bd['augmented']['implied_floor_deg']:.1f}°.\nSame assumption predicts "
               f"{bd['plain']['rss_prediction_sideslip_deg']:.1f}° for plain; observed "
               f"{bd['plain']['observed_sideslip_deg']:.1f}° (inside CI)."), y=-.19)

    # C ループ利得
    ax = fig.add_subplot(gs[0, 2])
    sw = g['travel_gain_sweep']
    K = [s['K_loop'] for s in sw]
    ax.plot(K, [s['median_deg'] for s in sw], 'o-', color=TRAV, lw=2.0, ms=7, mec='white', mew=1.2)
    ax.plot(K, [s['linear_median_deg'] for s in sw], color=DIM, lw=1.2, ls=(0, (4, 3)))
    ax.set_xscale('log')
    ax.set_xticks([0.5, 1, 2.6, 5, 10, 20]); ax.set_xticklabels(['0.5', '1', '2.6', '5', '10', '20'])
    kaug = 2.6 * (0.060 + 0.703)
    ax.axvline(kaug, color=TRAV, ls=':', lw=1.2)
    ax.text(kaug * 1.06, 17.5, T(f'追加モデルの\nループ利得 ≈ {kaug:.2f}', f'augmented\nloop gain ≈ {kaug:.2f}'),
            fontsize=8, color=INK)
    ax.text(12, 6.2, T('線形理論', 'linear theory'), fontsize=8, color=DIM)
    ax.set_ylim(0, 21)
    ax.set_xlabel(T('ループ利得 K（rad/s、対数）', 'loop gain K (rad/s, log)'), fontsize=9)
    ax.set_ylabel(T('理想 travel servo の誤差（度）', 'ideal travel-servo error (deg)'), fontsize=9)
    clean(ax)
    title(ax, T('C  理想でも 7.8° 残る — 利得の問題', 'C  Even ideal leaves 7.8° — a gain issue'))
    note(ax, T('横滑りは τ = 4 秒で揺らぐ。一次ループは速い揺らぎに\n追いつけず、その遅れは利得を上げるほど小さくなる。',
               'Sideslip fluctuates with τ = 4 s. A first-order loop lags\nfast fluctuations; the lag shrinks with gain.'), y=-.19)

    fig.suptitle(T('横滑りの誤差の内訳：制御則だけで 9.5°（うち定常成分 3°）、残りは制御則の外',
                   'Error budget: the law alone gives 9.5° (3° of it static); the rest lies outside the law'),
                 fontsize=13, weight='bold', x=.012, ha='left', y=.975)
    fig.text(.012, .895, T('実験 4 と同じ課題（OU 横滑り σ = 55°、τ = 4 秒、|φ| ≤ 60°、同じ乱数種）を、'
                           '当てはめた制御則だけで回した値。回路は通していない。',
                           'Same task as experiment 4 (OU sideslip σ = 55°, τ = 4 s, |φ| ≤ 60°, same seed), '
                           'run with the fitted control law only — no circuit.'),
             fontsize=9, color=DIM)
    save(fig, 'fig24_budget')


# ======================================================================== 図 25

def fig25(d):
    fig = plt.figure(figsize=(13.8, 5.3))
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(1, 3, wspace=.34, left=.05, right=.985, top=.80, bottom=.22)

    # A 前段：フェーザ和
    ax = fig.add_subplot(gs[0, 0])
    H, phi = np.radians(30), np.radians(25)
    p = np.radians([45, -45, -135, 135])
    A = np.maximum(0, np.cos(phi - p))
    tot = 0j
    for k, (pp, aa) in enumerate(zip(p, A)):
        z = aa * np.exp(1j * (H + pp))
        ax.annotate('', xy=(z.real, z.imag), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='-|>', color=HEAD if aa > 0 else GRID, lw=1.8))
        tot += z
    ax.annotate('', xy=(tot.real, tot.imag), xytext=(0, 0),
                arrowprops=dict(arrowstyle='-|>', color=TRAV, lw=3.0))
    ax.plot([0, np.cos(H) * 1.25], [0, np.sin(H) * 1.25], color=DIM, ls=':', lw=1.2)
    ax.text(np.cos(H) * 1.3, np.sin(H) * 1.3, 'H', color=DIM, fontsize=10, style='italic')
    ax.text(tot.real * 1.08, tot.imag * 1.08 + .05, 'H + φ', color=TRAV, fontsize=10.5, weight='bold')
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, ec=GRID, lw=1))
    ax.set_xlim(-1.3, 1.5); ax.set_ylim(-1.3, 1.5); ax.set_aspect('equal')
    ax.axis('off')
    title(ax, T('A  前段：掛け算で角度を足す', 'A  Stage 1: multiply to add angles'))
    note(ax, T(f"Σ_g [cos(φ − p_g)]₊ · e^(i(H + p_g)) = e^(i(H + φ))\n"
               f"全周で位相誤差 {d['C_phasor']['max_phase_err_deg']:.0e}°、大きさ 1（4 基底・90° 間隔）",
               f"Σ_g [cos(φ − p_g)]₊ · e^(i(H + p_g)) = e^(i(H + φ))\n"
               f"max phase error {d['C_phasor']['max_phase_err_deg']:.0e}°, magnitude 1 (4 bases, 90° apart)"),
         y=-.05)

    # B 後段：左右の入力
    ax = fig.add_subplot(gs[0, 1])
    gam = np.arange(12) * 2 * np.pi / 12
    G, Tt = np.radians(40), np.radians(0)
    L = (np.cos(G - gam) + np.cos(Tt - gam - np.pi / 2)) ** 2
    R = (np.cos(G - gam) + np.cos(Tt - gam + np.pi / 2)) ** 2
    x = np.arange(12)
    ax.bar(x - .2, L, width=.38, color=HEAD, label=T('左 PFL3（δ = +90°）', 'left PFL3 (δ = +90°)'))
    ax.bar(x + .2, R, width=.38, color=TRAV, label=T('右 PFL3（δ = −90°）', 'right PFL3 (δ = −90°)'))
    ax.set_xticks(x); ax.set_xticklabels([f'{int(round(np.degrees(g)))}' for g in gam], fontsize=7.6)
    ax.set_xlabel(T('細胞の位相 γ（度）', 'cell phase γ (deg)'), fontsize=9)
    ax.set_ylabel(T('補助入力 [cos(G−γ) + cos(T−γ−δ)]²', 'aux input [cos(G−γ) + cos(T−γ−δ)]²'), fontsize=8.6)
    ax.legend(fontsize=8, frameon=False, loc='upper right')
    clean(ax); ax.grid(axis='x', visible=False)
    ax.set_ylim(0, 4.6)
    title(ax, T('B  後段：二乗で角度を引く', 'B  Stage 2: square to subtract angles'))
    note(ax, T(f'G − T = 40° の例。γ で平均すると 1 + cos(G − T + δ) だけが残る。\n'
               f'左の和 {L.sum():.2f}、右の和 {R.sum():.2f} → (R−L)/(R+L) = {(R.sum()-L.sum())/(R.sum()+L.sum()):.3f} = sin 40°',
               f'Example G − T = 40°. Averaging over γ leaves 1 + cos(G − T + δ).\n'
               f'Left sum {L.sum():.2f}, right sum {R.sum():.2f} → (R−L)/(R+L) = {(R.sum()-L.sum())/(R.sum()+L.sum()):.3f} = sin 40°'),
         y=-.16)

    # C 出力
    ax = fig.add_subplot(gs[0, 2])
    dd = np.linspace(-180, 180, 73)
    u = [st.pfl3_comparator(np.radians(v), 0.0)[0] for v in dd]
    lin = [st.pfl3_comparator(np.radians(v), 0.0, square=False)[1] for v in dd]
    ax.plot(dd, np.sin(np.radians(dd)), color=DIM, lw=4, alpha=.35, label='sin(G − T)')
    ax.plot(dd, u, 'o', ms=4.5, color=TRAV, label=T('二乗あり：(R−L)/(R+L)', 'squared: (R−L)/(R+L)'))
    ax.plot(dd, lin, color=HEAD, lw=2, ls=(0, (5, 3)), label=T('二乗なし：左の和（恒等的に 0）', 'no square: left sum (≡ 0)'))
    ax.set_xlim(-180, 180); ax.set_xticks([-180, -90, 0, 90, 180]); ax.set_ylim(-1.25, 1.35)
    ax.set_xlabel('G − T' + T('（度）', ' (deg)'), fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc='upper left')
    clean(ax)
    title(ax, T('C  12 細胞の離散和で厳密', 'C  Exact for 12 discrete cells'))
    note(ax, T(f"最大誤差 {d['D_comparator']['max_abs_err_squared']:.0e}。二乗を外すと G − T の情報が消える。\n"
               '二乗は自己積なので、前段と同じく「掛け算」が要る。',
               f"max error {d['D_comparator']['max_abs_err_squared']:.0e}. Without the square, G − T vanishes.\n"
               'A square is a self-product: stage 2 also needs a multiplication.'), y=-.16)

    fig.suptitle(T('欠けていたのは二つの双線形演算：角度の足し算（前段）と引き算（後段）',
                   'The missing pieces are two bilinear operations: angle addition (stage 1) and subtraction (stage 2)'),
                 fontsize=13, weight='bold', x=.012, ha='left', y=.975)
    fig.text(.012, .895, T('理想化した式の数値確認（scripts/servo_theory.py の C・D 節）。回路は通していない。',
                           'Numerical check of the idealised formulas (sections C and D of scripts/servo_theory.py). No circuit.'),
             fontsize=9, color=DIM)
    save(fig, 'fig25_ops')


# ======================================================================== 図 26

def fig26(d):
    fig = plt.figure(figsize=(13.8, 5.2))
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(1, 3, wspace=.34, left=.06, right=.985, top=.80, bottom=.22)

    # A ランプ追従
    ax = fig.add_subplot(gs[0, 0])
    r = d['E_pll']['rows']
    xs = np.linspace(0, .999, 200)
    ax.plot(xs, np.degrees(np.arcsin(xs)), color=DIM, lw=4, alpha=.35, label='arcsin((dφ/dt)/K)')
    lk = [q for q in r if q['rate_over_K'] < 1]
    ax.plot([q['rate_over_K'] for q in lk], [abs(q['mean_err_deg']) for q in lk], 'o',
            ms=8, color=TRAV, mec='white', mew=1.3, label=T('シミュレーション', 'simulation'))
    ax.axvspan(1, 2.1, color=BAD, alpha=.07)
    ax.text(1.05, 80, T('ロック外れ\n（周回）', 'lock lost\n(cycle slips)'), fontsize=8.6, color=BAD, weight='bold')
    for q in r:
        if q['rate_over_K'] >= 1:
            ax.text(q['rate_over_K'], 8, T(f"{q['cycle_slips']:.0f} 周", f"{q['cycle_slips']:.0f} slips"),
                    ha='center', fontsize=8.2, color=INK)
    ax.set_xlim(0, 2.1); ax.set_ylim(0, 95)
    ax.set_xlabel(T('横滑りの変化率 / ループ利得  (dφ/dt)/K', 'sideslip rate / loop gain  (dφ/dt)/K'), fontsize=9)
    ax.set_ylabel(T('定常の遅れ（度）', 'steady lag (deg)'), fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc='upper left')
    clean(ax)
    title(ax, T('A  回り続ける横滑りへの追従', 'A  Tracking a rotating sideslip'))

    # B 時系列
    ax = fig.add_subplot(gs[0, 1])
    for frac, col, lab in ((0.5, TRAV, T('dφ/dt = 0.5K：ロック', 'dφ/dt = 0.5K: locked')),
                           (1.2, THIRD, T('dφ/dt = 1.2K：周回', 'dφ/dt = 1.2K: slipping'))):
        Kl = st.K_PLANT
        H, ph, es = 0.0, 0.0, []
        for _ in range(int(8 / st.DT)):
            T_ = H + ph
            H += Kl * np.sin(st.wrap(0 - T_)) * st.DT
            ph += frac * Kl * st.DT
            es.append(np.degrees(st.wrap(0 - (H + ph))))
        tt = np.arange(len(es)) * st.DT
        es = np.array(es)
        es[np.abs(np.diff(es, prepend=es[0])) > 180] = np.nan
        ax.plot(tt, es, color=col, lw=1.8, label=lab)
    ax.axhline(-30, color=DIM, ls=':', lw=1.0)
    ax.text(4.3, -22, 'arcsin 0.5 = 30°', fontsize=7.8, color=DIM, va='bottom')
    ax.set_xlim(0, 8); ax.set_ylim(-185, 185); ax.set_yticks([-180, -90, 0, 90, 180])
    ax.set_xlabel(T('時間（秒）', 'time (s)'), fontsize=9)
    ax.set_ylabel('e = G − T' + T('（度）', ' (deg)'), fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc='upper right')
    clean(ax)
    title(ax, T('B  ロックと周回', 'B  Lock and cycle slip'))

    # C 世界固定風
    ax = fig.add_subplot(gs[0, 2])
    rw = d['F_worldwind']['rows']
    b = [q['beta'] for q in rw]
    fr = [q['reached_fraction'] * 100 for q in rw]
    for bb, ff, q in zip(b, fr, rw):
        if q['monotone']:
            ax.plot(bb, ff, 'o', ms=10, color=TRAV, mec='white', mew=1.4)
        else:
            ax.plot(bb, ff, 'o', ms=10, mfc='white', mec=BAD, mew=2)
        ax.text(bb, ff - 3.2, f'{ff:.0f}%', ha='center', fontsize=8.2, color=INK)
    ax.axvline(1, color=DIM, ls=':', lw=1.2)
    ax.text(1.03, 72, T('β = 1：\n1 + ∂φ/∂H が\n0 に触れる', 'β = 1:\n1 + ∂φ/∂H\ntouches 0'),
            fontsize=8.2, color=DIM)
    ax.set_xlim(0, 2.2); ax.set_ylim(60, 104)
    ax.set_xlabel(T('風の強さ β（φ = β·sin(w − H)）', 'wind strength β (φ = β·sin(w − H))'), fontsize=9)
    ax.set_ylabel(T('5° 以内に到達した目標の割合（%）', 'goals reached within 5° (%)'), fontsize=9)
    clean(ax)
    title(ax, T('C  世界に固定された風', 'C  World-fixed wind'))
    note(ax, T('塗り：単調（1 + ∂φ/∂H > 0）、白抜き：非単調。\nβ = 0.95 の 99% は収束が遅いため（境界近く）。',
               'filled: monotone (1 + ∂φ/∂H > 0), hollow: not.\n99% at β = 0.95 reflects slow convergence near the boundary.'),
         y=-.16)

    fig.suptitle(T('一次 PLL として読むと、ロック範囲と符号条件が決まる',
                   'Read as a first-order PLL: lock range and sign condition'),
                 fontsize=13, weight='bold', x=.012, ha='left', y=.975)
    fig.text(.012, .895, T('理想の travel servo（K_H = 0、K_T = 1、K = 2.6 rad/s）。scripts/servo_theory.py の E・F 節。',
                           'Ideal travel servo (K_H = 0, K_T = 1, K = 2.6 rad/s). Sections E and F of scripts/servo_theory.py.'),
             fontsize=9, color=DIM)
    save(fig, 'fig26_pll')


def main():
    global LANG
    d = load('servo_theory.json')
    for LANG in ('ja', 'en'):
        fig22()
        fig23(d)
        fig24(d)
        fig25(d)
        fig26(d)


if __name__ == '__main__':
    main()
