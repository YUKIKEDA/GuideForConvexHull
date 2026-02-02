# 第8章 Chan's Algorithm（最適アルゴリズム）

Chan's Algorithm（チャンのアルゴリズム）は、凸包の頂点数 $h$ に依存する **$O(n \log h)$** を達成する、2次元凸包の**出力敏感**（output-sensitive）な最適アルゴリズムです。1996年に Timothy M. Chan によって発表されました。Graham Scan と Gift Wrapping のアイデアを組み合わせ、$h$ が小さいときに従来アルゴリズムより効率的になります。

---

## 8.1 O(n log h) の意義

### h が小さいときの効率化

| アルゴリズム | 計算量 | 特徴 |
|--------------|--------|------|
| Graham Scan / Andrew's Monotone Chain | $O(n \log n)$ | $h$ に依存しない |
| Gift Wrapping | $O(nh)$ | $h$ が小さいと速いが、最悪 $O(n^2)$ |
| **Chan's Algorithm** | $O(n \log h)$ | $h$ が小さいほど有利 |

凸包の頂点数 $h$ は $n$ より小さく、多くの応用では $h \ll n$ になります。例えば、円周上に一様に分布した点では $h = n$ ですが、矩形内に一様に分布した点では $h$ は $O(\log n)$ 程度になることが知られています。

$h \ll n$ のとき、$O(n \log h)$ は $O(n \log n)$ より高速です。また、凸包を求める問題は $\Omega(n \log h)$ の下界があるため、Chan's Algorithm は**理論的に最適**です。

---

## 8.2 アイデア：Graham Scan と Gift Wrapping の組み合わせ

Chan's Algorithm は、次の2つのアルゴリズムを組み合わせます。

1. **Graham Scan**：小さい点集合に対して凸包を効率的に求める（$O(m \log m)$、$m$ は点の数）
2. **Gift Wrapping**：凸包の頂点を1つずつ「巻いていく」ように求める（1頂点あたり $O(n)$ の素朴版、または効率化版）

**核心的なアイデア**

- 点集合を**大きさ $m$ のグループ**に分割し、各グループの凸包を Graham Scan で求める。
- これらの小さい凸包たちを「パッケージ」とみなし、Gift Wrapping で**パッケージ群の外側**を巻いていく。
- 各ステップで「次の凸包の頂点」を探す際、各小凸包への**接線**を求める。凸多角形への接線は**二分探索**で $O(\log m)$ で求まる。
- パラメータ $m$ を適切に選ぶと、全体で $O(n \log h)$ になる。

---

## 8.3 パラメータ m の選び方

$h$ は事前に分からないため、$m$ を**段階的に大きくしながら試行**します。

### 試行列

$m = 2^{2^t}$（$t = 1, 2, 3, \ldots$）とします。

- $t = 1$：$m = 4$
- $t = 2$：$m = 16$
- $t = 3$：$m = 256$
- $t = 4$：$m = 65536$
- …

### 成功条件

Gift Wrapping で $h$ 個の頂点を列挙し、スタートに戻ったら**成功**です。$h$ ステップを超えても戻らない場合、$m < h$ であり、小凸包に凸包の頂点が十分含まれていないため**失敗**とし、$m$ を大きくしてやり直します。

$m \geq h$ になったとき、必ず成功します。したがって、$m = 2^{2^t} \geq h$ となる最小の $t$ まで試行すればよく、$t = O(\log \log h)$ 回の試行で終了します。

---

## 8.4 完全なアルゴリズムの導出

### アルゴリズムの流れ

```
ChanHull(P):
    for t = 1, 2, 3, ...:
        m = 2^(2^t)
        if m > n:
            m = n

        // 1. 点を ceil(n/m) 個のグループに分割（各グループ最大 m 点）
        groups = P を m 個ずつに分割

        // 2. 各グループの凸包を Graham Scan で計算
        hulls = [ GrahamScan(g) for g in groups ]

        // 3. Gift Wrapping でマージ（最大 m ステップで打ち切り）
        result = JarvisMarchOnHulls(hulls, m)

        if result が成功（スタートに戻った）:
            return result
```

### Jarvis March on Hulls

通常の Gift Wrapping では、各ステップで「全点」から次の頂点を探します。Chan では、**各小凸包**に対して接線を求め、その接点のなかから「反時計回りに最も先にある点」を次の頂点とします。

- 現在の頂点を $p$ とする。
- 各小凸包 $H_i$ について、$p$ から $H_i$ への**右接線**（$p$ から見て $H_i$ に接し、$H_i$ を左に回る接線）の接点 $q_i$ を二分探索で求める。
- 全 $q_i$ のうち、$p$ から見て**反時計回りに最も先にある点**を次の凸包の頂点とする。

小凸包は凸多角形なので、接点の探索は二分探索で $O(\log m)$ です。グループ数は $\lceil n/m \rceil$ なので、1ステップあたり $O((n/m) \log m)$、全体で $O(h \cdot (n/m) \log m)$ となります。

### 接線の求め方（右接線）

凸多角形 $H$ の頂点が反時計回りに $v_0, v_1, \ldots, v_{k-1}$ と並んでいるとします。点 $p$ から $H$ への右接線の接点は、$\text{ccw}(p, v_i, v_{i+1})$ の符号が変わる境界を二分探索で見つけます。または、$p$ から見た極角が最小の頂点を接点とします。

### 計算量

- **Graham Scan 部分**：$\lceil n/m \rceil$ 個のグループ、各 $O(m \log m)$ → 合計 $O(n \log m)$
- **Jarvis 部分**：$h$ ステップ × $O((n/m) \log m)$ → $O(h \cdot (n/m) \log m)$

$m \geq h$ のとき、$h \cdot (n/m) \leq n$ なので、Jarvis 部分は $O(n \log m) = O(n \log h)$ です。Graham 部分も $O(n \log h)$。試行は $O(\log \log h)$ 回で、各試行のコストは $m$ に応じて増加するため、総和は $O(n \log h)$ にまとまります。

---

## 8.5 実装と検証

### Python 実装（簡略版）

```python
from typing import List, Optional, Tuple
import math

def cross(o, a, b) -> float:
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)

def ccw(o, a, b) -> int:
    c = cross(o, a, b)
    if c > 0: return 1
    if c < 0: return -1
    return 0

def dist_sq(a, b) -> float:
    return (b.x - a.x)**2 + (b.y - a.y)**2

def graham_scan(points: List[Point2D]) -> List[Point2D]:
    """Andrew's Monotone Chain で凸包を求める（簡略）"""
    if len(points) < 3:
        return list(points)
    points = sorted(points, key=lambda p: (p.x, p.y))
    def build(hull_points):
        hull = []
        for p in hull_points:
            while len(hull) >= 2 and cross(hull[-2], hull[-1], p) <= 0:
                hull.pop()
            hull.append(p)
        return hull
    lower = build(points)
    upper = build(reversed(points))
    return lower[:-1] + upper[:-1]

def right_tangent(p: Point2D, hull: List[Point2D]) -> int:
    """点 p から凸多角形 hull への右接線の接点のインデックス（二分探索）"""
    n = len(hull)
    if n == 1:
        return 0
    left, right = 0, n
    while right - left > 1:
        mid = (left + right) // 2
        if ccw(p, hull[mid], hull[(mid - 1) % n]) <= 0 and ccw(p, hull[mid], hull[(mid + 1) % n]) <= 0:
            return mid
        # 接点でない場合、どちら側に進むか判定
        a = ccw(p, hull[left], hull[mid])
        if a > 0:
            right = mid
        else:
            left = mid
    return left

def chan_hull(points: List[Point2D]) -> List[Point2D]:
    n = len(points)
    if n < 3:
        return list(points)

    t = 1
    while True:
        m = min(2 ** (2 ** t), n)
        if m < 3:
            t += 1
            continue

        # グループに分割
        hulls = []
        for i in range(0, n, m):
            group = points[i:i + m]
            if len(group) >= 3:
                hulls.append(graham_scan(group))
            elif len(group) == 2:
                hulls.append(group)
            elif len(group) == 1:
                hulls.append(group)

        if not hulls:
            t += 1
            continue

        # Gift Wrapping  on hulls
        start = min(points, key=lambda p: (p.x, p.y))
        hull = [start]
        current = start
        for _ in range(m):
            next_pt = None
            for h in hulls:
                if not h:
                    continue
                idx = right_tangent(current, h)
                q = h[idx]
                if next_pt is None or ccw(current, next_pt, q) > 0 or (ccw(current, next_pt, q) == 0 and dist_sq(current, q) > dist_sq(current, next_pt)):
                    next_pt = q
            if next_pt is None or next_pt == start:
                break
            hull.append(next_pt)
            current = next_pt

        if current == start and len(hull) >= 3:
            return hull
        t += 1
        if 2 ** (2 ** t) > n:
            return graham_scan(points)  # フォールバック
```

> 注：`right_tangent` の二分探索は、凸包の表現（反時計回り頂点列）に依存する。実装時は、接線の定義と境界条件を慎重に確認すること。

### 検証のポイント

1. ランダムな点集合で、Andrew's Monotone Chain の結果と一致するか確認する。
2. $h$ が小さい場合（例：円形や矩形内の一様分布）で、Graham Scan より高速になるか計測する。
3. 全点が凸包上（$h = n$）のケースで、正しく動作するか確認する。

---

## 8.6 正しさの証明（Chan's Algorithm の O(n log h) の証明スケッチ）

### 正しさ

- 各小凸包は Graham Scan で正しく求まっている。
- Gift Wrapping の各ステップで、現在の頂点から「全点の中で」反時計回りに最も先にある点を選べば、凸包の正しい次の頂点が得られる。
- Chan では、その「全点」を各小凸包の接点で代表している。各小凸包の接点は、その小凸包内で現在の頂点から反時計回りに最も先にある点である。したがって、全接点の中から反時計回りに最も先の点を選べば、全点の中での最良の点と一致する。
- $m \geq h$ のとき、凸包の全頂点が何らかの小凸包に含まれ（同じ点が複数グループに含まれる場合もあるが、凸包の頂点は必ず含まれる）、Gift Wrapping は $h$ ステップでスタートに戻る。よって、正しい凸包が得られる。

### O(n log h) の証明スケッチ

- 試行 $t$ では $m = 2^{2^t}$ とする。$m > n$ のときは $m = n$ とする。
- 成功するのは $m \geq h$ のとき。つまり $2^{2^t} \geq h$、$2^t \geq \log h$、$t \geq \log_2 \log h$ のとき。したがって試行回数は $O(\log \log h)$。
- 各試行のコストは $O(n \log m)$。成功時の $m$ は $\Theta(h)$ 程度なので、最後の成功試行のコストは $O(n \log h)$。
- それ以前の試行の総和は、$m = 4, 16, 256, \ldots$ と増えていく等比級数となり、$O(n \log h)$ で上から抑えられる。よって全体で $O(n \log h)$ となる。

---

## 本章のまとめ

- **Chan's Algorithm** は $O(n \log h)$ を達成する、2次元凸包の出力敏感な最適アルゴリズムである。
- **Graham Scan** で小グループの凸包を求め、**Gift Wrapping** でそれらをマージする。
- パラメータ $m$ を $2^{2^t}$ で増やしながら試行し、$m \geq h$ で成功する。
- $h$ が小さい場合に有効だが、実装はやや複雑で、$h$ が大きいときは Graham Scan や Andrew's Monotone Chain の方が単純で十分なことも多い。

次章では、**数値的ロバストネスと性能**を扱い、浮動小数点の落とし穴やベンチマークの考え方を学びます。
