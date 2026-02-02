# 第3章 Gift Wrapping（Jarvis March）

Gift Wrapping（ギフト包装法）は、凸包を求める最も直感的なアルゴリズムの1つです。**Jarvis March**（ジャービスの行進）とも呼ばれ、1973年に R.A. Jarvis によって発表されました。「最も左の点から始めて、ひたすら外側を巻いていく」という動作が、名前の由来（贈り物を包装するように点を包んでいく）にもなっています。

---

## 3.1 アルゴリズムのアイデア

### 「最も左の点から始めて、ひたすら外側を巻いていく」

Gift Wrapping の考え方は非常にシンプルです。

1. **スタート**：凸包の頂点として確実に含まれる点を1つ選ぶ。通常は **x 座標が最小**（同じなら y 座標が最小）の点を選ぶ。これが「最も左」の点であり、凸包の頂点の1つである。
2. **次の頂点を探す**：現在の頂点から、**すべての点を見て**「反時計回りに最も回った点」を次の凸包の頂点として選ぶ。
3. **繰り返す**：次の頂点がスタートの点に戻ったら終了。そうでなければ、その点を「現在の頂点」として 2 に戻る。

つまり、凸包の外周を**反時計回りに**1周するように、1頂点ずつ決めていきます。各ステップで「今いる頂点から見て、最も左に曲がった先の点」を選ぶことで、常に凸包の外側を進むことができます。

```
  ・ ・    ・
 ・   ・ ・  ・
・  P0  ―→ P1
 ・  ╲   ╱ ・     P0 から出発し、全点の中で
  ・ ・ P2 ・     「反時計回りに最も回った」点を
    ・ ・ ・      次々と選んでいく
```

---

## 3.2 疑似コードと動作原理

### 疑似コード

```
GiftWrapping(P):
    入力: 点集合 P = {p_1, ..., p_n}
    出力: 凸包の頂点列（反時計回り）

    if |P| < 3:
        return P   // 0, 1, 2 点の場合はそのまま返す

    // 最も左の点（x 最小、同点なら y 最小）を探す
    start = P のうち x 座標が最小の点（同点なら y 座標が最小）
    hull = [start]

    current = start
    repeat:
        next = P の任意の点（current 以外）
        for each point p in P:
            if p == current:
                continue
            // p が next より「反時計回りに先にある」なら next を更新
            if ccw(current, next, p) > 0:
                next = p
            else if ccw(current, next, p) == 0:
                // 同一直線上: より遠い点を選ぶ
                if distance(current, p) > distance(current, next):
                    next = p
        hull.append(next)
        current = next
    until current == start

    return hull
```

### 動作原理

- **ccw(current, next, p) > 0**  
  $p$ が $\overline{\text{current} \rightarrow \text{next}}$ の**左側**にある。つまり、`next` から見て `p` の方が反時計回りに先にある。したがって、`next` を `p` に更新する。
- **ccw(current, next, p) == 0**  
  3点が同一直線上。このときは、`current` から**より遠い**点を凸包の頂点として選ぶ。近い点を選ぶと、凸包上にあるべき点を飛ばしてしまう。
- **ccw(current, next, p) < 0**  
  $p$ が右側にある。`next` の方が反時計回りに先にあるので、`next` はそのまま。

初期の `next` は `current` 以外の任意の点で構いません。ループで全点と比較するため、最終的に「反時計回りに最も先にある点」が選ばれます。

---

## 3.3 計算量解析：O(nh)

$n$ を点数、$h$ を凸包の頂点数とします。

- **各頂点を決めるステップ**：全 $n$ 点と比較するので $O(n)$
- **頂点数**：凸包の頂点は $h$ 個
- **合計**：$O(n) \times h = O(nh)$

最悪の場合、すべての点が凸包上にある（$h = n$）とき、計算量は $O(n^2)$ になります。一方、凸包の頂点が少ない（$h \ll n$）場合、Graham Scan の $O(n \log n)$ より高速になることがあります。

| 状況 | 計算量 | 備考 |
|------|--------|------|
| 一般 | $O(nh)$ | $h$ は凸包の頂点数 |
| 最悪（$h = n$） | $O(n^2)$ | 全点が凸包上 |
| 最良（$h$ が小さい） | $O(n)$ に近い | 点が密集している場合 |

---

## 3.4 実装と演習

### Python 実装

```python
from typing import List
from math import sqrt

# 第1章・第2章で定義した Point2D, cross, ccw を使用
# ここでは簡略化のため同ファイル内で定義する場合の例

class Point2D:
    def __init__(self, x: float, y: float):
        self.x, self.y = x, y
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

def cross(o, a, b) -> float:
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)

def ccw(o, a, b) -> int:
    c = cross(o, a, b)
    if c > 0: return 1
    if c < 0: return -1
    return 0

def dist_sq(a: Point2D, b: Point2D) -> float:
    """距離の2乗（比較用、sqrt を避けて高速化）"""
    dx, dy = b.x - a.x, b.y - a.y
    return dx * dx + dy * dy

def gift_wrapping(points: List[Point2D]) -> List[Point2D]:
    """Gift Wrapping (Jarvis March) で凸包を求める。反時計回り。"""
    n = len(points)
    if n < 3:
        return list(points)

    # 最も左の点（x 最小、同点なら y 最小）
    start = min(points, key=lambda p: (p.x, p.y))
    hull = [start]
    current = start

    while True:
        # current 以外の点から next の候補を1つ選ぶ
        next_pt = None
        for p in points:
            if p == current:
                continue
            if next_pt is None:
                next_pt = p
                continue
            c = ccw(current, next_pt, p)
            if c > 0:
                next_pt = p
            elif c == 0:
                if dist_sq(current, p) > dist_sq(current, next_pt):
                    next_pt = p
        hull.append(next_pt)
        current = next_pt
        if current == start:
            break

    return hull[:-1]  # 最後の start の重複を除く
```

### C++ 実装

```cpp
#include <vector>
#include <algorithm>
using namespace std;

struct Point2D { double x, y; };

double cross(Point2D o, Point2D a, Point2D b) {
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

double distSq(Point2D a, Point2D b) {
    double dx = b.x - a.x, dy = b.y - a.y;
    return dx*dx + dy*dy;
}

vector<Point2D> giftWrapping(vector<Point2D> points) {
    int n = points.size();
    if (n < 3) return points;

    int startIdx = 0;
    for (int i = 1; i < n; i++) {
        if (points[i].x < points[startIdx].x ||
            (points[i].x == points[startIdx].x && points[i].y < points[startIdx].y))
            startIdx = i;
    }

    vector<Point2D> hull;
    int current = startIdx;

    do {
        hull.push_back(points[current]);
        int next = (current == 0) ? 1 : 0;
        for (int i = 0; i < n; i++) {
            if (i == current) continue;
            double c = cross(points[current], points[next], points[i]);
            if (c > 0)
                next = i;
            else if (c == 0 && distSq(points[current], points[i]) > distSq(points[current], points[next]))
                next = i;
        }
        current = next;
    } while (current != startIdx);

    return hull;
}
```

### 演習

1. 上記のコードを入力し、ランダムな点集合で凸包が正しく求まるか確認する。
2. 全点が同一直線上に並ぶケースで動作を確認する。
3. 重複点がある場合の挙動を確認し、必要なら前処理で重複除去を入れる。

---

## 3.5 利点・欠点の整理

### 利点

| 項目 | 説明 |
|------|------|
| **直感的** | アイデアが簡単で、実装も理解しやすい |
| **実装が単純** | 外積と距離の比較だけで実装できる |
| **$h$ が小さいとき有利** | 凸包の頂点が少ないと $O(nh)$ は $O(n \log n)$ より速い |
| **3次元への拡張が容易** | 同じ考え方で 3D 凸包にも使える |

### 欠点

| 項目 | 説明 |
|------|------|
| **最悪 $O(n^2)$** | 全点が凸包上だと遅い |
| **出力依存** | 凸包の頂点数 $h$ に計算量が依存する |
| **数値誤差に敏感** | 共線判定で eps の扱いを誤ると、同一直線上の点の選択を間違える |

---

## 3.6 正しさの証明（Gift Wrapping が必ず外側を回る理由）

Gift Wrapping が正しい凸包を返すことを、次の2点から説明します。

### （1）選ばれる点はすべて凸包の頂点である

各ステップで、`current` から「反時計回りに最も先にある点」を `next` として選んでいます。  
これは、`current` を端点とする凸包の辺の「もう一方の端点」を選んでいることに相当します。

凸包の定義より、凸包の任意の辺 $\overline{AB}$ に対して、$P$ のすべての点はその左側（または辺上）にあります。したがって、$A$ から反時計回りに最も先にある点は、凸包上の点であり、かつ $A$ に隣接する頂点です。よって、`next` は必ず凸包の頂点になります。

### （2）凸包の全頂点が漏れなく訪れる

凸包は凸多角形なので、頂点を反時計回りにたどると1周して戻ります。  
各ステップで「反時計回りに最も先にある点」を選ぶことは、凸包の**隣接する次の頂点**を選ぶことと同値です。したがって、スタートから始めて、凸包の頂点を反時計回りに1つずつ進み、スタートに戻るまでに凸包のすべての頂点がちょうど1回ずつ訪れます。

（同一直線上に複数の点がある場合は、その中で最も遠い点を選ぶことで、凸包の頂点として適切な点が選ばれます。近い点だけを選ぶと、凸包上の頂点を飛ばしてしまうことに注意。）

### まとめ

- 各ステップで選ぶ点は、凸包の頂点である。
- 反時計回りに隣接頂点をたどっているので、凸包の全頂点が重複・欠落なく得られる。

以上より、Gift Wrapping は正しい凸包を出力します。

---

## 本章のまとめ

- **Gift Wrapping**は、「最も左の点から始めて、反時計回りに最も曲がった点を順に選ぶ」アルゴリズムである。
- 計算量は $O(nh)$（$h$ は凸包の頂点数）。$h$ が小さいときは高速だが、最悪 $O(n^2)$ になる。
- 同一直線上にある点では、より遠い点を選ぶ必要がある。
- アイデアが単純で実装しやすい一方、凸包の頂点が多い場合は遅くなる。

次章では、$O(n \log n)$ で安定して動作する **Graham Scan** を学びます。
