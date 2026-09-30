#!/usr/bin/env python3
"""経路積分: 実測コネクトームがホームベクトルを保持できるかを測る。

コネクトームの 187,137 本は一切いじらない。学習するのは
  (1) 自己運動を入れる強さ（スカラ数個）
  (2) 集団活動 → ホームベクトル の線形読み出し（リッジ回帰の閉形式）
だけで、「この配線に経路積分の情報が載るか」を問う実験になっている。

対照実験（--baselines）で、結果が本当に配線由来かを確かめる:
  自己運動入力を切る / 配線をシャッフル / EPG だけから読む / hDelta の再帰を切る

使い方:
  python3 scripts/train_pi.py                 # 標準の実験
  python3 scripts/train_pi.py --baselines     # 対照実験つき
  python3 scripts/train_pi.py --gain-search   # 入力ゲインを探索
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from cxnet_np import CXNetworkNP, ROLE  # noqa: E402
from pi_task import (  # noqa: E402
    SelfMotionInput, random_walk, home_errors, ridge_fit, ridge_apply,
    heading_only_baseline, heading_speed_baseline, oracle_baseline,
)

MODEL = os.path.join(ROOT, 'model', 'flybrain-cx.gguf')
DT = 0.02
BURN_IN = 60          # 最初の 1.2 秒はバンプが立つまでの助走なので捨てる
SUBSAMPLE = 4         # 0.08 秒ごとに記録する


def population_sets(net):
    """読み出し候補の集団。どこにホームベクトルが載るのかを比べる。"""
    fb = np.concatenate([net.by_role[k] for k in
                         ('PFN', 'HDELTA', 'VDELTA', 'FC', 'FS', 'FR', 'PFR')])
    return {
        'hDelta': net.hdelta,
        'hDelta+vDelta': np.concatenate([net.hdelta, net.vdelta]),
        'PFN': net.pfn,
        'FB全体': np.sort(fb),
        'EPG のみ': net.epg,
        '全細胞': np.arange(net.N),
    }


def make_traj(args, seed):
    return random_walk(args.steps, dt=DT, rng=np.random.default_rng(seed),
                       sideslip_sigma_deg=args.sideslip,
                       sideslip_tau=getattr(args, 'sideslip_tau', 4.0))


def run_trials(net, smi, args, n_traj, seed0, clarity=1.0,
               self_motion=True, record=None, progress=''):
    """軌跡を回して、集団活動とホームベクトルを集める。"""
    X, Y, meta, bounds = [], [], [], []
    rec = np.arange(net.N) if record is None else record
    t0 = time.time()
    for k in range(n_traj):
        traj = make_traj(args, seed0 + k)
        net.reset(1 + k)
        acts, homes = [], []
        for t in range(args.steps):
            net.clear_drive()
            net.set_visual_scene(traj['th'][t], clarity)
            if self_motion:
                # set_visual_scene が LNO/SpsP も er_baseline にするので、その後で上書きする
                # 横滑りは左右の駆動比の違いとして入る（ここが本実験の肝）
                smi.apply(net.clamp_val, traj['v'][t], traj['beta'][t])
            net.step(DT)
            if t >= BURN_IN and (t - BURN_IN) % SUBSAMPLE == 0:
                acts.append(net.r[rec].copy())
                homes.append(traj['home'][t])
        X.append(np.asarray(acts, dtype=np.float32))
        Y.append(np.asarray(homes, dtype=np.float64))
        meta.append(traj)
        bounds.append(len(acts))
        if progress and (k + 1) % 20 == 0:
            print(f'    {progress} {k+1}/{n_traj}  ({time.time()-t0:.0f}s)', flush=True)
    return np.concatenate(X), np.concatenate(Y), (meta, bounds)


def leaky_filter(X, bounds, taus=(1.0, 3.0, 10.0, 30.0)):
    """記録した集団活動に、軌跡ごとに漏れ積分をかける。

    「回路の中の積分器（hDelta）が詰まっているのか、それとも
    そもそも方向の情報が無いのか」を切り分けるための外付け積分器。
    これで成績が上がるなら、情報はあって積分だけが失敗していることになる。
    """
    dt = DT * SUBSAMPLE
    out = np.zeros((X.shape[0], X.shape[1] * len(taus)), dtype=np.float32)
    alphas = [np.exp(-dt / t) for t in taus]
    p = 0
    for n in bounds:
        seg = X[p:p + n]
        accs = [np.zeros(X.shape[1]) for _ in taus]
        for k in range(n):
            for j, a in enumerate(alphas):
                accs[j] = a * accs[j] + seg[k] * dt
                out[p + k, j * X.shape[1]:(j + 1) * X.shape[1]] = accs[j]
        p += n
    return out


def evaluate(net, smi, args, tag='', **kw):
    """学習 → 評価を 1 回分やる"""
    Xtr, Ytr, _tr_meta = run_trials(net, smi, args, args.train, 1000,
                                    progress=f'{tag}学習用' if tag else '学習用', **kw)
    Xte, Yte, meta = run_trials(net, smi, args, args.test, 7000,
                                progress=f'{tag}評価用' if tag else '評価用', **kw)
    (mtr, btr) = _tr_meta
    (mte, bte) = meta
    out = {}
    for name, idx in population_sets(net).items():
        # run_trials は全細胞を記録しているので、ここで部分集合を取る
        W = ridge_fit(Xtr[:, idx], Ytr, alpha=args.alpha)
        pred = ridge_apply(W, Xte[:, idx])
        out[name] = home_errors(pred, Yte)
        out[name]['n_cells'] = int(len(idx))
    # 外付けの積分器を噛ませた版（回路内の積分器が詰まっているかの切り分け）
    if getattr(args, 'external_integrator', False):
        for name in ('PFN', 'hDelta'):
            idx = population_sets(net)[name]
            Ftr = leaky_filter(Xtr[:, idx], btr)
            Fte = leaky_filter(Xte[:, idx], bte)
            W = ridge_fit(Ftr, Ytr, alpha=args.alpha)
            out[f'{name}＋外付け積分'] = home_errors(ridge_apply(W, Fte), Yte)
            out[f'{name}＋外付け積分']['n_cells'] = int(Ftr.shape[1])
    return out, (Xtr, Ytr, Xte, Yte, meta)


def show(title, res):
    print(f'\n--- {title} ---')
    print(f'{"読み出す集団":16s} {"細胞数":>6s} {"向きの誤差(中央値, 95%CI)":>26s} '
          f'{"45°以内":>8s} {"R² (95%CI)":>22s}')
    for k, v in res.items():
        ci = v.get('angle_median_ci', [float('nan')] * 2)
        rci = v.get('r2_ci', [float('nan')] * 2)
        print(f'{k:16s} {v["n_cells"]:6d} {v["angle_median_deg"]:13.1f}° '
              f'[{ci[0]:5.1f},{ci[1]:5.1f}] {v["angle_frac_within_45"]*100:7.0f}% '
              f'{v["r2"]:8.3f} [{rci[0]:6.3f},{rci[1]:6.3f}]')


def measure_integration_tau(net, smi):
    """hDelta の実効時定数。短いパルスを入れて、活動がどれだけ保たれるか。"""
    net.reset(1)
    for t in range(200):                       # まず落ち着かせる
        net.clear_drive(); net.set_visual_scene(0.0, 1.0)
        smi.apply(net.clamp_val, 0.0)
        net.step(DT)
    base = net.r[net.hdelta].copy()
    for t in range(50):                        # 1 秒だけ「歩く」
        net.clear_drive(); net.set_visual_scene(0.0, 1.0)
        smi.apply(net.clamp_val, 1.0)
        net.step(DT)
    peak = net.r[net.hdelta].copy() - base
    amp0 = float(np.abs(peak).sum())
    trace = []
    for t in range(600):                       # 止まったあと、どれだけ残るか
        net.clear_drive(); net.set_visual_scene(0.0, 1.0)
        smi.apply(net.clamp_val, 0.0)
        net.step(DT)
        trace.append(float(np.abs(net.r[net.hdelta] - base).sum()) / max(amp0, 1e-9))
    trace = np.array(trace)
    below = np.where(trace < 1 / np.e)[0]
    tau = (below[0] + 1) * DT if len(below) else float('inf')
    return tau, trace


def baseline_control(args, fn):
    """ネットワークを通さない対照デコーダ。fn が軌跡から特徴量を作る。"""
    def feats(n_traj, seed0):
        F, Y = [], []
        for k in range(n_traj):
            traj = make_traj(args, seed0 + k)
            f = fn(traj)
            sel = slice(BURN_IN, args.steps, SUBSAMPLE)
            F.append(f[sel]); Y.append(traj['home'][sel])
        return np.concatenate(F), np.concatenate(Y)
    Ftr, Ytr = feats(args.train, 1000)
    Fte, Yte = feats(args.test, 7000)
    W = ridge_fit(Ftr, Ytr, alpha=args.alpha)
    res = home_errors(ridge_apply(W, Fte), Yte)
    res['n_cells'] = Ftr.shape[1]
    return res


def trajectory_stats(args):
    """課題の性質（止まっている割合、横滑りの大きさ）を確認用に出す。"""
    b, stopped = [], []
    for k in range(min(args.test, 10)):
        t = make_traj(args, 7000 + k)
        b.append(np.degrees(np.abs(t['beta'])))
        stopped.append(np.mean(t['v'] == 0))
    b = np.concatenate(b)
    return {'sideslip_abs_median_deg': float(np.median(b)),
            'sideslip_abs_p90_deg': float(np.percentile(b, 90)),
            'stopped_frac': float(np.mean(stopped))}


def direction_tracking(net, smi, args, n_traj=6):
    """PFN / hDelta の集団ベクトルは、方位を追うのか進行方向を追うのか。

    横滑りを入れた課題での中心的な問い。左右 PFN の ±66° ずれたベクトル基底が
    働いているなら、集団ベクトルは体の向き（方位）ではなく
    **実際に進んでいる向き（方位＋横滑り）** に寄るはず。
    どちらに近いかを、円周上の中央絶対誤差で比べる。
    """
    ph = net.phase
    groups = {'PFN': net.pfn, 'hDelta': net.hdelta, 'EPG': net.epg}
    acc = {k: {'th': [], 'travel': []} for k in groups}
    for k in range(n_traj):
        traj = make_traj(args, 31000 + k)
        net.reset(1 + k)
        for t in range(args.steps):
            net.clear_drive()
            net.set_visual_scene(traj['th'][t], 1.0)
            smi.apply(net.clamp_val, traj['v'][t], traj['beta'][t])
            net.step(DT)
            if t < BURN_IN or t % SUBSAMPLE or traj['v'][t] <= 0:
                continue          # 止まっている間は進行方向が定義できない
            for name, idx in groups.items():
                sel = idx[ph[idx] > -8]
                a = net.r[sel]
                if a.sum() <= 1e-9:
                    continue
                ang = np.arctan2((a * np.sin(ph[sel])).sum(), (a * np.cos(ph[sel])).sum())
                acc[name]['th'].append(ang - traj['th'][t])
                acc[name]['travel'].append(ang - traj['travel'][t])
    out = {}
    for name, d in acc.items():
        row = {}
        for ref in ('th', 'travel'):
            if not d[ref]:
                continue
            e = np.asarray(d[ref])
            # 各集団は基準から一定のオフセットを持つので、それを除いてばらつきを見る
            off = np.arctan2(np.mean(np.sin(e)), np.mean(np.cos(e)))
            resid = np.degrees(np.abs((e - off + np.pi) % (2 * np.pi) - np.pi))
            row[ref] = {'offset_deg': float(np.degrees(off)),
                        'median_abs_resid_deg': float(np.median(resid)),
                        'circ_r': float(np.hypot(np.mean(np.cos(e)), np.mean(np.sin(e))))}
        out[name] = row
    return out


def apply_integrator(net, hd_gain, hd_tau):
    """hDelta を積分器にするための、細胞側パラメータ 2 つ。

    コネクトームは一切いじらない。ここで変えるのは
      - hDelta の行ゲイン倍率: 既定の動作点だと hDelta は閾値下で完全に沈黙し、
        自己運動を強めて発火させると今度は PFN の方位同調が壊れる。その谷間を通す。
      - hDelta の時定数: 既定は 50 ms。README の「限界」に書いたとおり時定数は
        実測値ではなく仮置きで、積分器になるには秒オーダーが要る。
    どちらも配線図には書かれていない量なので、課題から決めるのが筋になる。
    """
    net.row_gain = net.row_gain.copy()
    net.row_gain[net.hdelta] *= hd_gain
    net.tau = net.tau.copy()
    net.tau[net.hdelta] = hd_tau
    return net


def shuffle_connectome(net, seed=0):
    """行ごとに入力元をシャッフルする（各細胞の入力本数と重み分布は保つ）。

    「配線の相手が意味を持っているか」を問うための対照。次数分布は同じなので、
    差が出れば『誰と誰が繋がっているか』が効いていることになる。
    """
    rng = np.random.default_rng(seed)
    W = net.W.tocsr(copy=True)
    for i in range(net.N):
        a, b = W.indptr[i], W.indptr[i + 1]
        if b > a:
            W.indices[a:b] = rng.choice(net.N, size=b - a, replace=False)
    W.has_sorted_indices = False
    W.sort_indices()
    net.W = W
    return net


GLOM_COL = re.compile(r'_([LR])(\d+)_C(\d+)')
NPZ = os.path.join(ROOT, 'data', 'cx_network.npz')


def _pfn_glomeruli(net, npz_path=NPZ):
    """PFN 各細胞の (細胞型, PB 側, 糸球体番号) を細胞名から取り出す。

    糸球体番号は GGUF には入っていない（位相として畳み込まれている）ので、
    抽出時の npz にある細胞名から引く。名前は `PFNd(PB04)_R6_C4` の形。
    """
    d = np.load(npz_path, allow_pickle=True)
    name_of = {int(b): str(n) for b, n in zip(d['body_id'], d['names'])}
    out = {}
    for i in net.pfn:
        m = GLOM_COL.search(name_of.get(int(net.body_id[i]), ''))
        if m:
            out[int(i)] = (net.type_names[net.type_id[i]], m.group(1), int(m.group(2)))
    return out


def break_pb_fb_mirror(net, npz_path=NPZ, verbose=True):
    """右 PFN の投射先を反転し、PB→FB の鏡像写像を壊す対照。

    実測では 左 PB は「列 = 糸球体 − 1」、右 PB は「列 = 10 − 糸球体」と
    **鏡像**になっていて、これが左右 PFN の約 66° の書き込みずれを作っている
    （＝進行方向を表すベクトル基底）。そこで右 PFN だけを、細胞型ごとに
    糸球体順に並べて**順序を反転**させる。すると右も左と同じ向きの写像になり、
    左右のずれが消える。

    配線シャッフルより狙いを絞った対照になっている点が肝で、
      - 各細胞の入力も重みも位相も、出力の本数も、一切変えていない
      - 変えたのは「右 PFN のどの細胞がどの列へ投射するか」だけ
    なので、差が出れば **この写像そのもの** が効いていることになる。
    """
    glom = _pfn_glomeruli(net, npz_path)
    groups = {}
    for i, (t, side, g) in glom.items():
        if side == 'R':
            groups.setdefault(t, []).append((g, i))
    perm = np.arange(net.N)
    n_moved = 0
    for t, items in groups.items():
        items.sort()                       # 糸球体の昇順
        ids = [i for _, i in items]
        for a, b in zip(ids, reversed(ids)):
            perm[a] = b                    # a の出力を b の出力に差し替える
        n_moved += sum(1 for a, b in zip(ids, reversed(ids)) if a != b)
    # 列 j = ニューロン j からの出力。perm[a]=b で「a の出力を b のものにする」
    net.W = net.W[:, perm].tocsr()
    if verbose:
        print(f'  PB→FB 鏡像を破壊: 右 PFN {n_moved} 細胞の投射先を入れ替え '
              f'（{len(groups)} 細胞型）')
    return net


def rotate_pfn_write(net, frac, side=+1, npz_path=NPZ, verbose=False):
    """片側の PFN が扇状体へ書き込む位相を、円周の frac だけ回す対照。

    `break_pb_fb_mirror` が「鏡像かどうか」の二択だったのに対し、こちらは
    左右のずれの**大きさ**を連続的に振るためのもの。細胞型ごとに右 PFN を
    位相順に並べ、リストを frac×N だけ巡回シフトして出力を入れ替える。
    各細胞の入力・重み・位相・出力本数はどれも変わらず、
    「どの細胞がどの列へ書くか」だけが回る。

    これで「実測の約 66〜70° というずれが、ほんとうに効いている値なのか」を
    用量反応として測れる（0° に近づければ基底が消えるはず）。
    """
    glom = _pfn_glomeruli(net, npz_path)
    perm = np.arange(net.N)
    sd = {'L': -1, 'R': +1}
    moved = 0
    for t in sorted({v[0] for v in glom.values()}):
        ids = [i for i, (tt, ss, _g) in glom.items()
               if tt == t and sd[ss] == side]
        if len(ids) < 2:
            continue
        ids.sort(key=lambda i: net.phase[i])
        k = int(round(frac * len(ids))) % len(ids)
        for a, b in zip(ids, ids[k:] + ids[:k]):
            perm[a] = b
        moved += sum(1 for a, b in zip(ids, ids[k:] + ids[:k]) if a != b)
    net.W = net.W[:, perm].tocsr()
    if verbose:
        print(f'  PFN の書き込み位相を {frac*360:+.0f}° 回転: {moved} 細胞')
    return net


def pfn_lr_write_offset(net, target='vdelta', bins=16):
    """いま走っているネットワークの中で、左右 PFN が扇状体へ書き込む位相差を測る。

    data/pi_circuit.json と同じ量を、対照を適用した後の W から測り直すためのもの。
    鏡像を壊した後は、これがほぼ 0 になるはず。
    """
    ph = net.phase
    tgt = getattr(net, target)
    tgt = tgt[ph[tgt] > -8]
    W = net.W.tocsr()
    peaks = {}
    for side, sd in (('left', -1), ('right', +1)):
        src = net.pfn[(net.pb_side[net.pfn] == sd) & (ph[net.pfn] > -8)]
        acc = np.zeros(bins); cnt = np.zeros(bins)
        sub = np.asarray(W[np.ix_(tgt, src)].todense())
        for a, i in enumerate(tgt):
            k = ((ph[i] - ph[src]) % (2 * np.pi) / (2 * np.pi) * bins).astype(int) % bins
            np.add.at(acc, k, sub[a]); np.add.at(cnt, k, 1)
        prof = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0)
        pos = np.maximum(prof, 0)
        if pos.sum() <= 0:
            return None
        ang = (np.arange(bins) + 0.5) / bins * 2 * np.pi
        peaks[side] = np.degrees(np.arctan2((pos * np.sin(ang)).sum(),
                                            (pos * np.cos(ang)).sum()))
    d = (peaks['left'] - peaks['right'] + 180) % 360 - 180
    return {'left_peak_deg': round(float(peaks['left']), 1),
            'right_peak_deg': round(float(peaks['right']), 1),
            'offset_deg': round(float(d), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', type=int, default=60)
    ap.add_argument('--test', type=int, default=20)
    ap.add_argument('--steps', type=int, default=900, help='1 軌跡のステップ数 (×0.02秒)')
    ap.add_argument('--alpha', type=float, default=1.0, help='リッジ回帰の正則化')
    ap.add_argument('--lno-gain', type=float, default=0.08)
    ap.add_argument('--spsp-gain', type=float, default=0.08)
    # ↓ コネクトームには書かれていない細胞側のパラメータ。既定値は探索で決めた。
    #   hDelta を積分器にするために必要な 2 つで、これが本実験の「学習する量」。
    ap.add_argument('--hd-gain', type=float, default=10.0,
                    help='hDelta の行ゲイン倍率（既定の動作点では hDelta が閾値下で沈黙するため）')
    ap.add_argument('--hd-tau', type=float, default=10.0,
                    help='hDelta の時定数 [秒]（既定は 0.05 秒。積分にはこれを伸ばす必要がある）')
    ap.add_argument('--sideslip', type=float, default=0.0,
                    help='横滑りの大きさ [度, OU 過程の標準偏差]。0 なら進行方向＝方位')
    ap.add_argument('--sideslip-tau', type=float, default=4.0,
                    help='横滑りの持続時間 [秒]。短いと積分窓の中で打ち消し合って課題にならない')
    ap.add_argument('--external-integrator', action='store_true',
                    help='集団活動に外付けの漏れ積分をかけた読み出しも評価する')
    ap.add_argument('--baselines', action='store_true')
    ap.add_argument('--gain-search', action='store_true')
    ap.add_argument('-o', '--out', default=os.path.join(ROOT, 'data', 'pi_result.json'))
    args = ap.parse_args()

    print(f'モデル読み込み: {MODEL}')
    net = CXNetworkNP(MODEL)
    apply_integrator(net, args.hd_gain, args.hd_tau)
    smi = SelfMotionInput(net, lno_gain=args.lno_gain, spsp_gain=args.spsp_gain)
    print(f'  {net.N} ニューロン / {net.W.nnz} 結合')
    print(f'  LNO 左{len(net.lno_left)}/右{len(net.lno_right)}  '
          f'SpsP 左{len(net.spsp_left)}/右{len(net.spsp_right)}  '
          f'hDelta {len(net.hdelta)}  PFN {len(net.pfn)}')
    print(f'  1 軌跡 = {args.steps * DT:.0f} 秒, 学習 {args.train} 本 / 評価 {args.test} 本')
    ts = trajectory_stats(args)
    results_ts = ts
    print(f'  横滑り |β| 中央値 {ts["sideslip_abs_median_deg"]:.1f}° / '
          f'90%点 {ts["sideslip_abs_p90_deg"]:.1f}°、止まっている時間 {ts["stopped_frac"]*100:.0f}%')

    tau, trace = measure_integration_tau(net, smi)
    print(f'\nhDelta の実効時定数（歩行を止めてから活動が 1/e になるまで）: {tau:.2f} 秒')

    results = {'integration_tau_s': tau, 'params': vars(args), 'trajectory': results_ts}

    if args.gain_search:
        print('\n=== 自己運動ゲインの探索 ===')
        best = None
        for lg in [0.1, 0.2, 0.35, 0.6, 1.0]:
            for sg in [0.1, 0.2, 0.35, 0.6, 1.0]:
                smi.lno_gain, smi.spsp_gain = lg, sg
                res, _ = evaluate(net, smi, argparse.Namespace(
                    train=12, test=6, steps=args.steps, alpha=args.alpha), tag='')
                m = res['hDelta+vDelta']['angle_median_deg']
                print(f'  LNO={lg:4.2f} SpsP={sg:4.2f} -> 向きの誤差 {m:5.1f}°')
                if best is None or m < best[0]:
                    best = (m, lg, sg)
        print(f'  最良: LNO={best[1]} SpsP={best[2]} ({best[0]:.1f}°)')
        smi.lno_gain, smi.spsp_gain = best[1], best[2]
        results['best_gains'] = {'lno': best[1], 'spsp': best[2]}

    if args.sideslip > 0:
        print('\n=== 集団ベクトルは方位を追うか、進行方向を追うか ===')
        dt_ = direction_tracking(net, smi, args)
        results['direction_tracking'] = dt_
        print(f'{"集団":10s} {"方位からのばらつき":>20s} {"進行方向からのばらつき":>22s} {"どちらに近いか":>14s}')
        for name, row in dt_.items():
            if 'th' not in row or 'travel' not in row:
                continue
            a, b = row['th']['median_abs_resid_deg'], row['travel']['median_abs_resid_deg']
            who = '進行方向' if b < a - 1 else ('方位' if a < b - 1 else '差なし')
            print(f'{name:10s} {a:18.1f}° {b:20.1f}° {who:>14s}')

    print('\n=== 本番 ===')
    res, data = evaluate(net, smi, args)
    show('実測コネクトーム（自己運動あり）', res)
    results['main'] = res

    if args.baselines:
        print('\n=== 対照実験 ===')
        res_ho = baseline_control(args, heading_only_baseline)
        res_hs = baseline_control(args, heading_speed_baseline)
        res_or = baseline_control(args, oracle_baseline)
        show('対照0: 回路を通さないデコーダ', {
            '方位のみ': res_ho,
            '方位＋速度（横滑りを知らない）': res_hs,
            '進行方向＋速度（天井）': res_or,
        })
        results['heading_only'] = res_ho
        results['heading_speed'] = res_hs
        results['oracle'] = res_or
        res_nm, _ = evaluate(net, smi, args, tag='[自己運動なし]', self_motion=False)
        show('対照1: 自己運動入力を切る（LNO/SpsP を静止値のまま）', res_nm)
        results['no_self_motion'] = res_nm

        net_sh = CXNetworkNP(MODEL)
        apply_integrator(net_sh, args.hd_gain, args.hd_tau)
        shuffle_connectome(net_sh, seed=0)
        smi_sh = SelfMotionInput(net_sh, lno_gain=smi.lno_gain, spsp_gain=smi.spsp_gain)
        res_sh, _ = evaluate(net_sh, smi_sh, args, tag='[配線シャッフル]')
        show('対照2: 配線をシャッフル（入力本数は保つ）', res_sh)
        results['shuffled'] = res_sh

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f'\n結果: {args.out}')


if __name__ == '__main__':
    main()
