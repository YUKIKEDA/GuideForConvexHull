# 付録C 参考実装コード

本教材の各章で紹介した凸包アルゴリズムの**最小実装**への参照と、統合した参考実装をまとめます。詳細な解説は各章を参照してください。

---

## 章別の実装参照

| アルゴリズム | 章 | 主な実装箇所 |
|--------------|-----|--------------|
| 幾何プリミティブ（cross, ccw） | 第2章 | 2.1 外積による向き判定の実装 |
| Gift Wrapping | 第3章 | 3.4 実装と演習 |
| Graham Scan | 第4章 | 4.5 完全実装とデバッグ |
| Andrew's Monotone Chain | 第5章 | 5.4 実装の簡潔さと数値安定性 |
| 動的凸包（素朴版） | 第6章 | 6.3 応用：リアルタイム処理 |
| QuickHull | 第7章 | 7.5 実装演習 |
| Chan's Algorithm | 第8章 | 8.5 実装と検証 |
| 3D 幾何（signed_volume） | 第12章 | 12.1 3D 幾何の基礎 |
| 3D Gift Wrapping | 第13章 | 13.4 実装 |

---

## 統合参考実装（2次元）

`Appendix/code/convex_hull_2d.py` に、2次元凸包の主要アルゴリズムをまとめた参考実装を用意しています。以下にその概要を示します。

### 共通の幾何プリミティブ

```python
from dataclasses import dataclass
from typing import List

@dataclass
class Point2D:
    x: float
    y: float

def cross(o: Point2D, a: Point2D, b: Point2D) -> float:
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)

def ccw(o: Point2D, a: Point2D, b: Point2D) -> int:
    c = cross(o, a, b)
    return 1 if c > 0 else (-1 if c < 0 else 0)
```

### Andrew's Monotone Chain（推奨・最小実装）

```python
def andrew_monotone_chain(points: List[Point2D]) -> List[Point2D]:
    if len(points) < 3:
        return list(points)
    points = sorted(points, key=lambda p: (p.x, p.y))
    def build(hull_pts):
        hull = []
        for p in hull_pts:
            while len(hull) >= 2 and cross(hull[-2], hull[-1], p) <= 0:
                hull.pop()
            hull.append(p)
        return hull
    lower = build(points)
    upper = build(reversed(points))
    return lower[:-1] + upper[:-1]
```

### 利用方法

1. 各章の実装を参照して、アルゴリズムの動作を理解する
2. `Appendix/code/convex_hull_2d.py` をコピーし、プロジェクトに組み込む
3. 必要に応じて、eps の追加や退化ケースの拡張を行う

---

## 外部ライブラリでの凸包計算

### SciPy（Python）

```python
from scipy.spatial import ConvexHull
import numpy as np

points = np.array([[0, 0], [1, 0], [0.5, 1], [0.5, 0.5]])
hull = ConvexHull(points)
# hull.vertices: 凸包の頂点のインデックス
# hull.simplices: 2Dでは辺の頂点ペア、3Dでは面の頂点トリプル
```

### Qhull（コマンドライン）

```bash
# 2次元凸包
echo "2\n4\n0 0\n1 0\n0.5 1\n0.5 0.5" | qconvex n

# 3次元凸包（rbox でテストデータ生成）
rbox 10 D3 | qconvex s o TO result.off
```

---

## テスト用の最小スニペット

```python
# 動作確認用
points = [Point2D(0, 0), Point2D(1, 0), Point2D(1, 1), Point2D(0, 1), Point2D(0.5, 0.5)]
hull = andrew_monotone_chain(points)
print(len(hull))  # 4（正方形の4頂点、内部点は含まない）
```

```python
# convex_hull_2d.py
"""
2次元凸包 参考実装
本教材の付録Cで参照する統合実装です。
"""

from dataclasses import dataclass
from typing import List

# ========== 幾何プリミティブ ==========

@dataclass
class Point2D:
    x: float
    y: float


def cross(o: Point2D, a: Point2D, b: Point2D) -> float:
    """3点 O, A, B の外積。> 0: 左折, = 0: 共線, < 0: 右折"""
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)


def ccw(o: Point2D, a: Point2D, b: Point2D) -> int:
    """反時計回りなら 1, 共線なら 0, 時計回りなら -1"""
    c = cross(o, a, b)
    return 1 if c > 0 else (-1 if c < 0 else 0)


# ========== Andrew's Monotone Chain（推奨） ==========

def andrew_monotone_chain(points: List[Point2D]) -> List[Point2D]:
    """Andrew's Monotone Chain。O(n log n)。反時計回り。"""
    if len(points) < 3:
        return list(points)
    points = sorted(points, key=lambda p: (p.x, p.y))

    def build(hull_pts):
        hull = []
        for p in hull_pts:
            while len(hull) >= 2 and cross(hull[-2], hull[-1], p) <= 0:
                hull.pop()
            hull.append(p)
        return hull

    lower = build(points)
    upper = build(reversed(points))
    return lower[:-1] + upper[:-1]


# ========== Gift Wrapping ==========

def dist_sq(a: Point2D, b: Point2D) -> float:
    return (b.x - a.x) ** 2 + (b.y - a.y) ** 2


def gift_wrapping(points: List[Point2D]) -> List[Point2D]:
    """Gift Wrapping (Jarvis March)。O(nh)。反時計回り。"""
    if len(points) < 3:
        return list(points)
    start = min(points, key=lambda p: (p.x, p.y))
    hull = [start]
    current = start

    while True:
        next_pt = None
        for p in points:
            if p is current:
                continue
            if next_pt is None:
                next_pt = p
                continue
            c = ccw(current, next_pt, p)
            if c > 0:
                next_pt = p
            elif c == 0 and dist_sq(current, p) > dist_sq(current, next_pt):
                next_pt = p
        hull.append(next_pt)
        current = next_pt
        if current is start:
            break

    return hull[:-1]


# ========== Graham Scan ==========

import math


def graham_scan(points: List[Point2D]) -> List[Point2D]:
    """Graham Scan。O(n log n)。反時計回り。"""
    if len(points) < 3:
        return list(points)
    base = min(points, key=lambda p: (p.y, p.x))
    others = [p for p in points if p is not base]

    def cmp_key(p):
        angle = math.atan2(p.y - base.y, p.x - base.x)
        return (angle, dist_sq(base, p))

    others.sort(key=cmp_key)

    stack = [base]
    for p in others:
        while len(stack) >= 2 and ccw(stack[-2], stack[-1], p) <= 0:
            stack.pop()
        stack.append(p)

    return stack


# ========== QuickHull ==========

def quick_hull(points: List[Point2D]) -> List[Point2D]:
    """QuickHull。平均 O(n log n)、最悪 O(n^2)。反時計回り。"""
    if len(points) < 3:
        return list(points)
    points = sorted(points, key=lambda p: (p.x, p.y))
    left, right = points[0], points[-1]

    upper = [p for p in points if cross(left, right, p) > 0]
    lower = [p for p in points if cross(left, right, p) < 0]

    def find_hull(pts, a, b):
        if not pts:
            return []
        farthest = max(pts, key=lambda p: cross(a, b, p))
        s1 = [p for p in pts if p is not farthest and cross(a, farthest, p) > 0]
        s2 = [p for p in pts if p is not farthest and cross(farthest, b, p) > 0]
        return find_hull(s1, a, farthest) + [farthest] + find_hull(s2, farthest, b)

    hull_upper = find_hull(upper, left, right)
    hull_lower = find_hull(lower, right, left)

    return [left] + hull_upper + [right] + hull_lower


# ========== 使用例 ==========

if __name__ == "__main__":
    points = [
        Point2D(0, 0),
        Point2D(1, 0),
        Point2D(1, 1),
        Point2D(0, 1),
        Point2D(0.5, 0.5),
    ]
    hull = andrew_monotone_chain(points)
    print(f"Andrew: {len(hull)} vertices")  # 4

    hull2 = gift_wrapping(points)
    print(f"Gift Wrapping: {len(hull2)} vertices")  # 4

```