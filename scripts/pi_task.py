#!/usr/bin/env python3
"""経路積分課題: 中心複合体にホームベクトルを保持させる。

課題の定義
----------
ハエが開けた場所をランダムに歩く。各時刻で分かっているのは
自分の向き（視覚ランドマークから）と、前に進んだ量だけ。
このとき「出発点（巣）がいまどっちに、どれだけ離れているか」を
中心複合体の活動から読み出せるか、を問う。

  入力  : 視覚手がかり（真の方位）＋ 前進速度（LNO/SpsP へ）
  正解  : ホームベクトル (hx, hy) = 出発点 − 現在位置（アロセントリック座標）
  読み出し: hDelta などの集団活動からの線形デコード

何を学習し、何を凍結するか
--------------------------
凍結: コネクトーム 187,137 本の重みは一切触らない。
学習: (1) 自己運動をどれだけ強く入れるかのゲイン数個
      (2) 集団活動 → ホームベクトル の線形読み出し（リッジ回帰の閉形式）

つまり「実在の配線が経路積分を支持しているか」を測る実験であって、
配線を課題に合わせて作り変えるのではない。

配線の実測（scripts/measure_pi_circuit.py で確認できる）
--------------------------------------------------------
  LNO 出力の 85% と SpsP 出力の 81% が PFN へ入る（自己運動の入口）
  LNO は対側の PB 半球の PFN を抑制し、SpsP は同側を駆動する（左右分離）
  PB→FB 投射は左右で鏡像（L: 列=糸球体-1, R: 列=10-糸球体）
  そのため左右の PFN は扇状体へ約 66° ずれた位相で書き込む（PFNd→vDelta 実測）
  hDelta は入力の 26% を hDelta 自身から受ける（全部興奮性）＝積分器の素地
  hDelta → PFL3 の総重み 40.3 は FC → PFL3 の 20.6 より大きい（帰巣の出口）
"""

from __future__ import annotations

import numpy as np

TAU = 2 * np.pi


# ---------------------------------------------------------------- 軌跡
def random_walk(n_steps, dt=0.02, rng=None, speed=1.4, turn_sigma=2.2,
                turn_tau=0.35, speed_jitter=0.3, stop_go=True,
                walk_bout=1.5, stop_bout=1.5,
                sideslip_sigma_deg=0.0, sideslip_tau=4.0, sideslip_max_deg=60.0):
    """相関のあるランダムウォークを作る。

    角速度をオルンシュタイン・ウーレンベック過程にして、実際のハエのように
    「しばらく直進してときどき曲がる」軌跡にする。

    **stop_go が肝**: 速度を「歩く / 止まる」の 2 状態にして、止まっている間も
    向きは変わり続けるようにする。速度がほぼ一定だと、変位は方位の積分だけで
    ほぼ当たってしまい（∫v·dir dt ≈ v̄·∫dir dt）、自己運動の信号が要らない
    課題になってしまう。実際それで最初は「自己運動を切っても同じ成績」
    「配線をシャッフルしたほうが良い成績」という結果が出た。
    止まる時間を入れると、方位だけを積んだ答えは必ずずれるので、
    速度の信号を使えたかどうかが成績に効くようになる。

    Returns dict with th (方位), v (前進速度), pos (位置), home (ホームベクトル)
    """
    rng = rng or np.random.default_rng(0)
    omega = np.zeros(n_steps)
    a = np.exp(-dt / turn_tau)
    w = 0.0
    for k in range(n_steps):
        w = a * w + np.sqrt(1 - a * a) * turn_sigma * rng.standard_normal()
        omega[k] = w
    th = np.cumsum(omega * dt) % TAU

    if stop_go:
        # 歩く / 止まる の電信過程。各状態の継続時間は指数分布。
        v = np.zeros(n_steps)
        k = 0
        walking = rng.random() < 0.5
        while k < n_steps:
            mean_len = walk_bout if walking else stop_bout
            n = max(1, int(rng.exponential(mean_len / dt)))
            seg = slice(k, min(n_steps, k + n))
            if walking:
                amp = speed * (1 + speed_jitter * rng.standard_normal())
                v[seg] = max(0.0, amp)
            k += n
            walking = not walking
    else:
        v = np.clip(speed * (1 + speed_jitter * rng.standard_normal(n_steps)), 0, None)

    # 横滑り: 体の向き（方位）と、実際に進んでいる向きのずれ。
    # これを入れると travelling direction ≠ heading になり、
    # 「方位と速度だけ」からは原理的に変位を求められなくなる。
    # ハエにこれを解かせているのが、左右 PFN の ±66° ずれたベクトル基底のはず。
    if sideslip_sigma_deg > 0:
        beta = np.zeros(n_steps)
        a = np.exp(-dt / sideslip_tau)
        b = 0.0
        sig = np.radians(sideslip_sigma_deg)
        for k in range(n_steps):
            b = a * b + np.sqrt(1 - a * a) * sig * rng.standard_normal()
            beta[k] = b
        beta = np.clip(beta, -np.radians(sideslip_max_deg), np.radians(sideslip_max_deg))
    else:
        beta = np.zeros(n_steps)

    # 実際に進む向き = 方位 + 横滑り
    travel = th + beta
    step = (v * dt)[:, None] * np.stack([np.cos(travel), np.sin(travel)], axis=1)
    pos = np.cumsum(step, axis=0)
    home = -pos                      # 出発点は原点。ホームベクトル = 原点 − 現在地
    return {'th': th, 'omega': omega, 'v': v, 'beta': beta, 'travel': travel,
            'pos': pos, 'home': home, 'dt': dt}


def _leaky_integrals(vec, dt, taus):
    """時定数を複数用意した漏れ積分。ベースラインを不利にしないため。

    ネットワーク側は 189 次元を読み出すのに、ベースラインが 2 次元だと
    当てはめの自由度で負けているだけ、という反論が立つ。時定数を何種類か
    重ねて次元を増やし、「情報は同じだが表現は十分柔軟」な競争相手にする。
    """
    out = np.zeros((len(vec), 2 * len(taus)))
    accs = [np.zeros(2) for _ in taus]
    alphas = [np.exp(-dt / t) for t in taus]
    for k in range(len(vec)):
        for j, (a, acc) in enumerate(zip(alphas, accs)):
            accs[j] = a * acc + vec[k] * dt
            out[k, 2 * j:2 * j + 2] = accs[j]
    return out


TAUS = (1.0, 3.0, 10.0, 30.0)


def heading_only_baseline(traj, taus=TAUS):
    """対照A: 「速度を無視して方位だけを積む」デコーダ。

    速度が一定だとこれで解けてしまう（最初に踏んだ穴）。
    """
    th = traj['th']
    vec = np.stack([np.cos(th), np.sin(th)], axis=1)
    return _leaky_integrals(vec, traj['dt'], taus)


def heading_speed_baseline(traj, taus=TAUS):
    """対照B: 方位と速度を**完璧に**知っているが、横滑りは知らないデコーダ。

    これが本命の対照。∫v·(cosθ, sinθ)dt を計算しており、横滑りが無い課題なら
    これで正解に到達できる（＝ネットワークが勝てる余地はない）。
    横滑りを入れると、この特徴量からは原理的に変位が求まらなくなる。
    **ネットワークがこれを上回ったときだけ**、横滑りの情報が
    左右 PFN の駆動比を通って回路に入り、使われたと言える。
    """
    th, v = traj['th'], traj['v']
    vec = v[:, None] * np.stack([np.cos(th), np.sin(th)], axis=1)
    return _leaky_integrals(vec, traj['dt'], taus)


def oracle_baseline(traj, taus=TAUS):
    """上限: 進行方向（方位＋横滑り）も速度も知っている場合。

    課題自体が漏れ積分で解ける範囲にあるかを確認するための天井。
    """
    tr, v = traj['travel'], traj['v']
    vec = v[:, None] * np.stack([np.cos(tr), np.sin(tr)], axis=1)
    return _leaky_integrals(vec, traj['dt'], taus)


# ---------------------------------------------------------------- 自己運動の入れ方
class SelfMotionInput:
    """前進速度を LNO / SpsP に入れる。

    配線の実測に合わせて、左右で別々の値をクランプできるようにしてある。
    既定では左右同じ値（前進のみ、横滑りなし）を入れる。このとき
    左右の PFN は同じだけ駆動され、その和の向きは方位そのものになる
    ＝「速度 × 方位」が扇状体に書き込まれ、それを積むとホームベクトルになる。

    横滑り（sideslip）を入れると左右の駆動が非対称になり、進行方向が
    方位からずれる。PB→FB の左右 66° オフセットが効いてくるのはこの場合。
    """

    def __init__(self, net, lno_gain=0.35, spsp_gain=0.35, baseline=0.02, flow_axis_deg=45.0):
        self.net = net
        self.lno_gain = lno_gain
        self.spsp_gain = spsp_gain
        self.baseline = baseline
        self.flow_axis = np.radians(flow_axis_deg)

    def drive(self, v, sideslip=0.0, mode='normal'):
        """前進速度 v（と横滑り角）から、左右それぞれの自己運動信号を作る。

        左右の視覚流検出器が ±flow_axis 方向に向いているとみなし、
        進行方向（体の正面から sideslip だけずれた向き）を射影する。
        横滑り 0 なら左右とも v·cos(45°) で等しくなる。

        mode は機構を切り分けるための対照:
          'normal'    そのまま
          'equalized' 左右を平均値で置き換える。和（= 速さ）は保ったまま
                      差（= 横滑りの向き）だけを消す
          'swapped'   左右を入れ替える。差の符号だけが反転する
        """
        sL = v * np.cos(sideslip - self.flow_axis)
        sR = v * np.cos(sideslip + self.flow_axis)
        if mode == 'equalized':
            sL = sR = 0.5 * (sL + sR)
        elif mode == 'swapped':
            sL, sR = sR, sL
        elif mode != 'normal':
            raise ValueError(f'unknown drive mode: {mode}')
        return float(sL), float(sR)

    def apply(self, r_clamp, v, sideslip=0.0, mode='normal'):
        """クランプ値の配列に自己運動ぶんを書き込む（その場で書き換える）"""
        net = self.net
        sL, sR = self.drive(v, sideslip, mode)
        b = self.baseline
        for i in net.lno_left:
            r_clamp[i] = max(0.0, b + self.lno_gain * sL)
        for i in net.lno_right:
            r_clamp[i] = max(0.0, b + self.lno_gain * sR)
        for i in net.spsp_left:
            r_clamp[i] = max(0.0, b + self.spsp_gain * sL)
        for i in net.spsp_right:
            r_clamp[i] = max(0.0, b + self.spsp_gain * sR)
        return r_clamp


# ---------------------------------------------------------------- 評価
def bootstrap_ci(values, n_boot=2000, stat=np.median, seed=0):
    """ブートストラップで統計量の 95% 信頼区間を出す。

    「26.8° と 25.8°」のような差を誤差なしで比べても意味がないので、
    対照実験と比べるときは必ずこれを付ける。
    """
    rng = np.random.default_rng(seed)
    n = len(values)
    bs = [stat(values[rng.integers(0, n, n)]) for _ in range(n_boot)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def home_errors(pred, true, with_ci=True):
    """予測と正解のホームベクトルから、向きの誤差と距離の誤差を出す。

    向きだけが合っていればよい（帰る方角が分かる）場面と、距離も要る場面が
    あるので両方返す。向きの誤差は円周上で測る。
    """
    pa = np.arctan2(pred[:, 1], pred[:, 0])
    ta = np.arctan2(true[:, 1], true[:, 0])
    ang = np.degrees(np.abs((pa - ta + np.pi) % TAU - np.pi))
    pr = np.hypot(pred[:, 0], pred[:, 1])
    tr = np.hypot(true[:, 0], true[:, 1])
    # 距離は「巣から離れているとき」だけ意味があるので、近すぎる点は除く
    far = tr > np.percentile(tr, 20)
    rel = np.abs(pr[far] - tr[far]) / np.maximum(tr[far], 1e-6)
    r2 = 1 - np.sum((pred - true) ** 2) / np.sum((true - true.mean(axis=0)) ** 2)
    out = {
        'angle_median_deg': float(np.median(ang)),
        'angle_mean_deg': float(np.mean(ang)),
        'angle_frac_within_45': float(np.mean(ang < 45)),
        'dist_rel_median': float(np.median(rel)),
        'r2': float(r2),
    }
    if with_ci:
        lo, hi = bootstrap_ci(ang)
        out['angle_median_ci'] = [lo, hi]
        # R^2 のブートストラップは残差ごと再標本化する
        rng = np.random.default_rng(1)
        n = len(true); bs = []
        for _ in range(500):
            s_ = rng.integers(0, n, n)
            bs.append(1 - np.sum((pred[s_] - true[s_]) ** 2)
                      / np.sum((true[s_] - true[s_].mean(axis=0)) ** 2))
        out['r2_ci'] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    return out


def ridge_fit(X, Y, alpha=1.0):
    """閉形式のリッジ回帰。勾配法も PyTorch も要らない。

    X: (n_samples, n_features) 集団活動
    Y: (n_samples, 2) ホームベクトル
    """
    Xb = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    A = Xb.T @ Xb
    A[np.diag_indices_from(A)] += alpha
    W = np.linalg.solve(A, Xb.T @ Y)
    return W


def ridge_apply(W, X):
    Xb = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    return Xb @ W
