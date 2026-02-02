# 第10章 退化ケースとエッジケースの網羅

凸包アルゴリズムの実装では、**退化ケース**（degenerate case）や**エッジケース**（edge case）で失敗することが少なくありません。この章では、代表的な退化ケースへの対応方法と、テスト駆動開発による品質担保を体系的に扱います。

---

## 10.1 全点が同一直線上

### 現象

すべての点が1本の直線上に並んでいるとき、凸包は**線分**（両端の2点）になります。多角形ではなく線分になるため、多くのアルゴリズムで特別扱いが必要です。

### 各アルゴリズムでの影響

| アルゴリズム | 影響 | 対策 |
|--------------|------|------|
| **Graham Scan** | 極角がすべて同じでソートが不定。スタックで中間点が pop され、両端のみ残る想定。 | 同角では距離が近い点を先にソートする |
| **Andrew's Monotone Chain** | 上下凸包が同じ線分になる。重複を除く結合で正しく線分になる。 | 通常の実装でそのまま動作することが多い |
| **Gift Wrapping** | 同一直線上では「より遠い点」を選ぶ。両端から巻いていき、2点で戻る。 | 距離による tie-break を入れる |
| **QuickHull** | 直線 LR の上側・下側に点が存在しない。FindHull に渡される点が空になる。 | 空の場合の早期 return を正しく処理する |

### 実装のポイント

```python
# Graham Scan の場合：同角では距離が近い順
def sort_key(base, p):
    angle = math.atan2(p.y - base.y, p.x - base.x)
    dist_sq = (p.x - base.x)**2 + (p.y - base.y)**2
    return (angle, dist_sq)  # 近い点が先 → 遠い点が来たときに近い点を pop

# Gift Wrapping の場合：同一直線上では遠い点を選ぶ
if ccw(current, next_pt, p) == 0:
    if dist_sq(current, p) > dist_sq(current, next_pt):
        next_pt = p
```

---

## 10.2 重複点の処理

### 現象

同じ座標 $(x, y)$ に複数の点が存在する場合です。入力に重複があると、凸包の頂点として同じ点が何度も出力されたり、無限ループに陥ったりすることがあります。

### 対策

**前処理で重複除去する**

```python
def remove_duplicates(points: List[Point2D]) -> List[Point2D]:
    if not points:
        return []
    points = sorted(points, key=lambda p: (p.x, p.y))
    result = [points[0]]
    for i in range(1, len(points)):
        if (points[i].x, points[i].y) != (points[i-1].x, points[i-1].y):
            result.append(points[i])
    return result

# 凸包計算の前に呼ぶ
points = remove_duplicates(points)
hull = andrew_monotone_chain(points)
```

**重複を許容する場合**

- 凸包の頂点列を構築した後、連続する同一座標を除去する
- Gift Wrapping で「次の頂点」を選ぶ際、`current` と同一の点はスキップする

---

## 10.3 3点以下の入力

### 期待される動作

| 点数 | 凸包 | 出力 |
|------|------|------|
| 0 | 存在しない | 空リスト `[]` |
| 1 | その1点 | `[p]` |
| 2 | 2点を結ぶ線分 | `[p1, p2]` |
| 3 | 三角形または線分（共線時） | 3点または2点 |

### 実装のテンプレート

```python
def convex_hull(points: List[Point2D]) -> List[Point2D]:
    n = len(points)
    if n <= 1:
        return list(points)
    if n == 2:
        return list(points)  # 線分
    # 3点かつ共線の場合は線分（2点）を返す
    # 多くのアルゴリズムは 3 点以上を仮定しており、
    # 共線はスタックや再帰の過程で自然に処理される
    return _convex_hull_impl(points)
```

3点が同一直線上にある場合、凸包は線分（2点）です。Graham Scan や Andrew's Monotone Chain では、スタック操作で中間の点が pop され、両端の2点のみが残ります。

---

## 10.4 共線点を許容するか削除するかの設計判断

### 2つの方針

| 方針 | 説明 | 利点 | 欠点 |
|------|------|------|------|
| **共線点を含める** | 辺上にある点も凸包の頂点として出力する | 凸包の「境界上」の点をすべて得られる | 頂点数が増え、後処理が煩雑になることがある |
| **共線点を削除する** | 凸包の「角」にある点だけを頂点とする | 頂点数が少なく、多角形として扱いやすい | 境界上の情報が減る |

### 実装上の違い

**共線点を削除する（一般的）**

- Graham Scan / Andrew's Monotone Chain：`ccw <= 0` で pop すると、同一直線上では**より遠い点**が残り、中間点は除外される
- Gift Wrapping：同一直線上では最も遠い点を選ぶ

**共線点を含める**

- `ccw < 0` のときだけ pop し、`ccw == 0` のときは push する（ただし、ソート順や tie-break の設計に注意が必要）
- 辺上の点も頂点として出力される

用途に応じて、仕様を決めておくことが重要です。

---

## 10.5 テスト駆動開発と単体テスト

### 単体テストの設計

凸包アルゴリズムの単体テストでは、次の性質を検証します。

1. **凸包の頂点は入力点から選ばれている**
2. **凸包は反時計回り（または時計回り）に並んでいる**
3. **すべての入力点が凸包の内部または境界上にある**
4. **凸包の面積が非負で、期待と一致する**（既知の正解がある場合）

```python
def test_hull_contains_all_points(hull, points):
    """すべての点が凸包の内部または境界上にある"""
    for p in points:
        assert point_in_or_on_hull(p, hull), f"Point {p} outside hull"

def test_hull_is_ccw(hull):
    """凸包の頂点が反時計回りに並んでいる"""
    for i in range(len(hull)):
        a, b, c = hull[i], hull[(i+1)%len(hull)], hull[(i+2)%len(hull)]
        assert ccw(a, b, c) >= 0, "Hull is not CCW"
```

### テストデータ生成技法

#### ランダム点生成

```python
import random

def random_points_uniform(n, x_range=(0, 1), y_range=(0, 1)):
    """矩形内に一様分布"""
    return [Point2D(random.uniform(*x_range), random.uniform(*y_range)) for _ in range(n)]

def random_points_normal(n, center=(0.5, 0.5), std=0.2):
    """正規分布"""
    return [Point2D(random.gauss(center[0], std), random.gauss(center[1], std)) for _ in range(n)]

def random_points_on_circle(n, center=(0.5, 0.5), radius=0.4):
    """円周上（h = n の最悪ケースに近い）"""
    return [Point2D(center[0] + radius * math.cos(2*math.pi*i/n),
                    center[1] + radius * math.sin(2*math.pi*i/n)) for i in range(n)]
```

#### 最悪ケース生成（全点が凸包上）

```python
def points_all_on_hull(n):
    """全点が凸包上（円周上または凸多角形の頂点）"""
    return random_points_on_circle(n)
```

#### 退化ケース生成

```python
def points_collinear(n, start=(0, 0), end=(1, 1)):
    """同一直線上に並んだ点"""
    return [Point2D(
        start[0] + (end[0]-start[0]) * i / max(n-1, 1),
        start[1] + (end[1]-start[1]) * i / max(n-1, 1)
    ) for i in range(n)]

def points_with_duplicates(n, unique_ratio=0.7):
    """重複点を含む"""
    k = max(1, int(n * unique_ratio))
    base = random_points_uniform(k)
    return [random.choice(base) for _ in range(n)]
```

#### 大規模データ生成（100万点）

```python
def large_random_points(n=1_000_000):
    """100万点の一様乱数"""
    return random_points_uniform(n, x_range=(0, 1e9), y_range=(0, 1e9))
```

大規模データでは、メモリと実行時間に注意し、必要に応じてサンプリングや分割テストを行います。

### 既知の正解データとの比較

**Qhull を参照実装として使う**

```python
# Qhull は qhull コマンドまたは scipy.spatial.ConvexHull で利用可能
from scipy.spatial import ConvexHull
import numpy as np

def convex_hull_qhull(points):
    pts = np.array([(p.x, p.y) for p in points])
    hull = ConvexHull(pts)
    return [Point2D(pts[i, 0], pts[i, 1]) for i in hull.vertices]

def test_against_qhull(algorithm, n_trials=100):
    for _ in range(n_trials):
        points = random_points_uniform(random.randint(10, 1000))
        my_hull = algorithm(points)
        ref_hull = convex_hull_qhull(points)
        assert hulls_equivalent(my_hull, ref_hull, points)
```

**手計算の正解を使う**

```python
def test_simple_cases():
    # 正方形の4頂点
    points = [Point2D(0,0), Point2D(1,0), Point2D(1,1), Point2D(0,1)]
    hull = convex_hull(points)
    assert len(hull) == 4

    # 三角形
    points = [Point2D(0,0), Point2D(1,0), Point2D(0.5, 1)]
    hull = convex_hull(points)
    assert len(hull) == 3
```

### リグレッションテスト

過去にバグの原因となった入力と期待出力を記録し、修正後も同じ結果になることを確認します。

```python
# 過去のバグの再現ケース
REGRESSION_CASES = [
    {"points": [...], "expected_hull_size": 5},
    {"points": collinear_10_points, "expected_hull_size": 2},
]

def test_regression():
    for case in REGRESSION_CASES:
        hull = convex_hull(case["points"])
        assert len(hull) == case["expected_hull_size"]
```

---

## 本章のまとめ

- **全点が同一直線上**のときは凸包が線分になる。ソートや tie-break の設計で正しく2点を返す。
- **重複点**は前処理で除去するのが安全。除去しない場合は頂点列から重複を省く。
- **3点以下**は早期 return で明示的に扱う。
- **共線点**を凸包に含めるかは仕様次第。多くの実装では除外する。
- **テスト駆動開発**では、ランダム・最悪・退化・大規模などのテストデータを体系的に生成し、Qhull との比較やリグレッションテストで品質を保つ。

次章では、**アルゴリズム選択ガイド**を扱い、入力の特性に応じてどのアルゴリズムを選ぶかを学びます。
