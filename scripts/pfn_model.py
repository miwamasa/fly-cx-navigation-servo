#!/usr/bin/env python3
"""PFNd/PFNv → hΔB を「数理モデル」として最後まで書き下し、検証する。

  python3 scripts/pfn_model.py
  → data/pfn_model.json

これは何であって、何でないか
----------------------------
**これはコネクトームの測定ではない。** 文献が述べている描像
（docs/REFERENCE_PFN_HDB.md の 1〜6 章）を、こちらが数式として書き下した
**理想化モデル**である。配線から導いた値は 1 つも入っていない
（実測値を使う条件には `measured` と明記し、どこから読んだかを出力に残す）。

目的は 3 つ。

  1. 文献の描像が数式として**閉じている**ことを確かめる
     （身体座標の速度と頭方位から、世界座標の進行方向が厳密に出るか）
  2. その厳密性が**どの性質に依存しているか**を、1 つずつ壊して測る
  3. 第 5 試験（scripts/coord_transform.py の円環回帰）の**陽性対照**にする
     — 理想モデルが a=1, b=1 を返さないなら、採点系の方が壊れている

モデル
------
身体座標の進行方向を φ、頭方位を θ、速度を s とする。
4 集団 g ∈ {PFNd-L, PFNd-R, PFNv-L, PFNv-R} がそれぞれ選好方向 p_g を持ち、
その活動の振幅は身体座標速度の**半波整流した射影**とする。

    A_g = s · [cos(φ − p_g)]₊            （[x]₊ = max(0, x)）

各集団はコラム方向に頭方位 θ の正弦波として活動し、hΔB へは集団ごとの
**書き込みオフセット** Δ_g だけずらして投射する。hΔB はその和を受ける。

    V = Σ_g w_g · A_g · exp(i(θ + Δ_g))

読み出しは V の偏角 ψ = arg V。

命題
----
p_g が 90° 等間隔で、Δ_g = p_g、w_g がすべて等しいとき

    V = s · exp(i(θ + φ))

が**厳密に**成り立つ。振幅も φ に依らず s で一定である。
本スクリプトはこれを数値でも確かめる（`exactness`）。

系（走らせて初めて分かったこと）
--------------------------------
効いているのは Δ_g と p_g の**差**だけで、絶対値ではない。
Δ_g − p_g が 4 本で共通の定数なら、その定数は世界座標の原点をずらすだけで、
円環回帰の定数項 c に吸収される。したがって

  ・**入口が全部抑制性でも（4 本一様に 180° 回るだけなら）変換は壊れない**
  ・壊れるのは Δ_g − p_g が**集団ごとにばらつく**とき

である。最初は「抑制性の入口 → 選好の反転 → 破綻」という筋を想定して
条件を組んだが、一様な反転では a=1, b=1 のまま合格してしまった。
そこで `spread_*` 条件を足して、ばらつきの量に対する応答を測ることにした。
（この条件は結果を見たあとに追加したもので、事前に決めていたものではない。）

壊す条件
--------
measured_offset  Δ_g を実測の書き込みオフセットに（data/pfn_basis.json）
measured_weight  w_g を実測のシナプス総重みに
measured_conc    実測の集中度 R_g を基底の長さに掛ける（位相のばらつきの効果）
flipped          入口が抑制性の場合を模して p_g を一様に 180° 回す
spread_*         Δ_g − p_g を集団ごとにずらす（ばらつきの量を振る）
collapsed        実測の φ 選好（前方に潰れた +29/−42/+11/−45）を使う
pfnv_silent      PFNv の重みを 0 にする（本リポジトリの既定の動作点で起きていたこと）
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

from coord_transform import circular_fit  # noqa: E402  第 5 試験と同じ採点系

OUT = os.path.join(ROOT, 'data', 'pfn_model.json')
BASIS = os.path.join(ROOT, 'data', 'pfn_basis.json')
SHUNT = os.path.join(ROOT, 'data', 'shunting.json')

GROUPS = ('PFNd-L', 'PFNd-R', 'PFNv-L', 'PFNv-R')

# 文献が述べている選好方向（度）。これは仮定であって測定値ではない
PREF_LIT = {'PFNd-L': +45.0, 'PFNd-R': -45.0, 'PFNv-L': -135.0, 'PFNv-R': +135.0}

TAU = 2 * np.pi


def wrap(a):
    return (a + np.pi) % TAU - np.pi


# ---------------------------------------------------------------- モデル本体

def amplitudes(phi, speed, pref_deg):
    """A_g = s·[cos(φ − p_g)]₊ — 身体座標速度の半波整流した射影。"""
    return {g: float(speed * max(0.0, np.cos(phi - np.radians(p))))
            for g, p in pref_deg.items()}


def hdb_vector(theta, phi, speed, pref_deg, write_deg, weight, length):
    """V = Σ w_g · A_g · exp(i(θ + Δ_g)) — hΔB が受け取るフェーザの和。"""
    A = amplitudes(phi, speed, pref_deg)
    v = 0j
    for g in pref_deg:
        v += weight[g] * length[g] * A[g] * np.exp(1j * (theta + np.radians(write_deg[g])))
    return v, A


def sweep(pref_deg, write_deg, weight, length, step=45):
    """θ と φ を独立に振って、第 5 試験と同じ形の行を作る。"""
    rows = []
    for tdeg in range(0, 360, step):
        for pdeg in range(0, 360, step):
            for sp in (0.5, 1.0):
                v, _ = hdb_vector(np.radians(tdeg), np.radians(pdeg), sp,
                                  pref_deg, write_deg, weight, length)
                mag = float(abs(v))
                rows.append({'theta': tdeg, 'phi': pdeg, 'speed': sp, 'mag': mag,
                             'psi': float(np.angle(v)) if mag > 1e-12 else None})
    return rows


# ---------------------------------------------------------------- 厳密性の確認

def exactness(pref_deg, write_deg, weight, length, n=721):
    """理想条件で V = s·exp(i(θ+φ)) が厳密かを、細かい格子で確かめる。

    角度誤差の最大値と、振幅のリップル（max/min − 1）を返す。
    リップルは「4 本という数」の意味を示す量でもある。
    """
    err, mags = [], []
    for tdeg in (0.0, 37.0, 123.0, 271.0):
        th = np.radians(tdeg)
        for phi in np.linspace(0, TAU, n, endpoint=False):
            v, _ = hdb_vector(th, phi, 1.0, pref_deg, write_deg, weight, length)
            if abs(v) < 1e-12:
                err.append(np.pi); mags.append(0.0); continue
            err.append(abs(wrap(np.angle(v) - (th + phi))))
            mags.append(abs(v))
    mags = np.array(mags)
    return {'max_angle_err_deg': float(np.degrees(max(err))),
            'mag_min': float(mags.min()), 'mag_max': float(mags.max()),
            'ripple': float(mags.max() / mags.min() - 1.0) if mags.min() > 0 else None}


def ripple_vs_n(ns=(2, 3, 4, 5, 6, 8)):
    """選好方向を N 本等間隔にしたときの振幅リップル。

    半波整流でも利得が方向に依らないのは N がいくつからか、を数値で出す。
    「なぜ 4 本なのか」に対する、モデル側からの答え。
    """
    out = {}
    for N in ns:
        pref = {f'b{k}': 360.0 * k / N for k in range(N)}
        w = {g: 1.0 for g in pref}
        mags = []
        for phi in np.linspace(0, TAU, 721, endpoint=False):
            v, _ = hdb_vector(0.0, phi, 1.0, pref, pref, w, w)
            mags.append(abs(v))
        mags = np.array(mags)
        out[str(N)] = {'mag_min': float(mags.min()), 'mag_max': float(mags.max()),
                       'ripple': float(mags.max() / mags.min() - 1.0)
                       if mags.min() > 1e-12 else None}
    return out


# ---------------------------------------------------------------- 実測値の読み込み

def measured():
    """実測値（ここだけがコネクトーム由来）。どこから読んだかも一緒に返す。"""
    pb = json.load(open(BASIS, encoding='utf-8'))
    write = {g: float(np.degrees(pb['groups'][g]['write_offset'])) for g in GROUPS}
    conc = {g: float(pb['groups'][g]['R']) for g in GROUPS}
    wsum = {g: float(pb['groups'][g]['total_weight']) for g in GROUPS}
    m = max(wsum.values())
    weight = {g: wsum[g] / m for g in GROUPS}

    sh = json.load(open(SHUNT, encoding='utf-8'))
    row = next(r for r in sh['rows'] if r['name'] == '分流 f=1.0')
    collapsed = {g: float(row['probe'][g]['phi_pref_deg']) for g in GROUPS}
    return {'write_offset_deg': write, 'concentration': conc,
            'weight_norm': weight, 'total_weight': wsum,
            'collapsed_pref_deg': collapsed,
            'source': {'write_offset_deg': 'data/pfn_basis.json',
                       'concentration': 'data/pfn_basis.json',
                       'weight_norm': 'data/pfn_basis.json',
                       'collapsed_pref_deg': 'data/shunting.json（分流 f=1.0）'}}


# ---------------------------------------------------------------- 条件

def conditions(mes):
    one = {g: 1.0 for g in GROUPS}
    lit = dict(PREF_LIT)
    W, C, N = mes['write_offset_deg'], mes['concentration'], mes['weight_norm']

    return [
        # name, 選好 p_g, 書き込み Δ_g, 重み w_g, 基底の長さ（集中度）, 説明
        ('理想（文献どおり）', lit, lit, one, one,
         'p=Δ=±45/±135、重みも集中度も等しい。命題そのもの'),
        ('実測の書き込みオフセット', lit, W, one, one,
         'Δ だけ実測（+30/−22/−146/+141）に替える'),
        ('＋実測の重み', lit, W, N, one,
         'さらに w をシナプス総重みに。PFNv は PFNd の 1/5 しかない'),
        ('＋実測の集中度', lit, W, N, C,
         'さらに基底の長さに R を掛ける。PFNd の R は 0.2 以下'),
        ('入口が抑制性（一様に 180° 反転）', {g: v + 180.0 for g, v in lit.items()},
         lit, one, one,
         '4 本とも同じだけ回るので、定数項に吸収されて壊れない'),
    ] + [
        (f'Δ−p のばらつき ±{s:.0f}°', lit,
         {g: lit[g] + s * f for g, f in zip(GROUPS, (+1, -1, -1, +1))}, one, one,
         '書き込みオフセットと選好の差を集団ごとにずらす')
        for s in (15, 30, 45, 60, 90)
    ] + [
        ('選好が前方に潰れた場合', mes['collapsed_pref_deg'], lit, one, one,
         '実測の φ 選好（+29/−42/+11/−45）を使う'),
        ('PFNv が沈黙（＝PFNd の 2 本だけ）', lit, lit,
         {'PFNd-L': 1.0, 'PFNd-R': 1.0, 'PFNv-L': 0.0, 'PFNv-R': 0.0}, one,
         '本リポジトリの既定の動作点で起きていたこと。2 本では後方が表せない'),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--out', default=OUT)
    a = ap.parse_args()

    mes = measured()
    one = {g: 1.0 for g in GROUPS}

    # 命題の数値確認
    ex = exactness(PREF_LIT, PREF_LIT, one, one)
    print('=== 命題の確認（理想条件）')
    print(f"  角度誤差の最大 {ex['max_angle_err_deg']:.2e} 度")
    print(f"  振幅 {ex['mag_min']:.6f}〜{ex['mag_max']:.6f}（リップル {ex['ripple']:.2e}）")

    rv = ripple_vs_n()
    print('\n=== 何本必要か（半波整流での振幅リップル）')
    for n, v in rv.items():
        r = v['ripple']
        print(f"  {n} 本: リップル {'—' if r is None else f'{r*100:6.1f}%'}"
              f"  （振幅 {v['mag_min']:.3f}〜{v['mag_max']:.3f}）")

    # 条件ごとに第 5 試験の採点系へ
    print(f"\n=== 第 5 試験（同じ円環回帰）に通す")
    print(f"{'条件':32s} {'方位 a':>8s} {'進行方向 b':>10s} {'残差 R':>8s} {'Δ−p ばらつき':>7s}  判定")
    out = []
    crit = json.load(open(os.path.join(ROOT, 'data', 'coord_transform.json'),
                          encoding='utf-8'))['criteria']
    for name, pref, write, weight, length, why in conditions(mes):
        rows = sweep(pref, write, weight, length)
        fit = circular_fit(rows)
        ok = bool(fit
                  and crit['heading_gain'][0] <= fit['heading_gain'] <= crit['heading_gain'][1]
                  and crit['travel_gain'][0] <= fit['travel_gain'] <= crit['travel_gain'][1]
                  and fit['residual_R'] >= crit['residual_R_min'])
        mags = [r['mag'] for r in rows if r['speed'] == 1.0]
        # Δ−p のばらつき（円環の標準偏差）— 系が述べるとおり、効くのはこの量
        d = np.radians([write[g] - pref[g] for g in pref])
        R_d = float(np.hypot(np.cos(d).mean(), np.sin(d).mean()))
        rec = {'name': name, 'why': why, 'fit': fit, 'pass': ok,
               'pref_deg': pref, 'write_deg': write, 'weight': weight, 'length': length,
               'delta_minus_pref_deg': {g: float(wrap(np.radians(write[g] - pref[g]))
                                                * 180 / np.pi) for g in pref},
               'delta_spread_deg': float(np.degrees(np.sqrt(-2 * np.log(R_d))))
               if R_d > 1e-9 else None,
               'mag_min': float(min(mags)), 'mag_max': float(max(mags)),
               'n_silent': sum(1 for r in rows if r['psi'] is None)}
        out.append(rec)
        sp = rec['delta_spread_deg']
        sp_s = '—' if sp is None else f'{sp:5.0f}°'
        if fit:
            print(f"{name:32s} {fit['heading_gain']:+8.2f} {fit['travel_gain']:+10.2f} "
                  f"{fit['residual_R']:8.2f} {sp_s:>7s}  {'合格' if ok else '不合格'}")
        else:
            print(f"{name:32s} {'—':>8s} {'—':>10s} {'—':>8s} {sp_s:>7s}  沈黙")

    res = {'note': 'これはコネクトームの測定ではなく、文献の描像を書き下した理想化モデル。'
                   '実測値を使う項目は measured に出典を明記してある。',
           'pref_literature_deg': PREF_LIT, 'criteria': crit,
           'exactness': ex, 'ripple_vs_n': rv, 'measured': mes, 'conditions': out}
    with open(a.out, 'w', encoding='utf-8') as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(f'\n→ {a.out}')


if __name__ == '__main__':
    main()
