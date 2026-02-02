# 第20章 スクラッチ実装プロジェクト

これまでの章で学んだ内容を総合し、凸包ライブラリを**スクラッチで設計・実装**するプロジェクトを扱います。API 設計、2D/3D 実装、ベンチマーク、可視化、最適化、ドキュメントとテストまで、実践的な指針をまとめます。

---

## 20.1 2D 凸包ライブラリの設計

### API 設計

**基本的な API**

```python
# 凸包計算（アルゴリズムを選択可能）
def convex_hull_2d(points: List[Point2D], algorithm: str = "andrew") -> List[Point2D]:
    """
    points: 入力点のリスト
    algorithm: "andrew" | "graham" | "gift_wrapping" | "quickhull"
    returns: 凸包の頂点列（反時計回り）
    """

# 点が凸包の内部または境界上にあるか
def point_in_hull(p: Point2D, hull: List[Point2D]) -> bool:
    ...

# 凸包の面積
def hull_area(hull: List[Point2D]) -> float:
    ...
```

**拡張 API**

```python
# 複数アルゴリズムの統一インターフェース
class ConvexHull2D:
    def compute(self, points: List[Point2D]) -> List[Point2D]: ...
    def verify(self, hull: List[Point2D], points: List[Point2D]) -> bool: ...
```

### データ構造

- **Point2D**：`(x, y)` を持つクラスまたは namedtuple
- **凸包**：頂点のリスト。反時計回りで一貫させる
- **オプション**：頂点のインデックスだけを返すか、頂点のコピーを返すかを設計で決める

### ディレクトリ構成の例

```
convex_hull/
├── __init__.py
├── point.py        # Point2D, Point3D
├── geom.py         # cross, ccw, signed_volume
├── hull2d/
│   ├── __init__.py
│   ├── andrew.py
│   ├── graham.py
│   ├── gift_wrapping.py
│   └── quickhull.py
├── hull3d/
│   └── ...
├── visualize.py
└── tests/
```

---

## 20.2 3D 凸包の完全実装

### 実装のステップ

1. **幾何プリミティブ**：`signed_volume`、法線ベクトル（第12章）
2. **初期四面体**：4点が共面でないことを確認（第16章）
3. **増分構築**：点の追加、見える面の削除、ホライゾンへの新面追加（第14章）
4. **データ構造**：簡易的な面・辺の管理、または Half-edge / DCEL（第15章）

### 最小構成

まずは、面を「3頂点のタプル」のリストで表現する簡易実装から始めます。隣接関係は持たず、可視性判定とホライゾンの計算を素朴に実装します。動作を確認したうえで、Half-edge を導入して効率化する、という段階的アプローチが扱いやすいです。

### 実装のチェックリスト

- [ ] 4点以下の入力
- [ ] 全点が共面
- [ ] 退化した四面体（4点が同一平面上）
- [ ] ランダムな点集合で Qhull と結果を比較
- [ ] 法線の向きが一貫しているか確認

---

## 20.3 ベンチマークと Qhull との比較

### ベンチマークの設計

**計測対象**

- 各アルゴリズムの実行時間
- メモリ使用量（大規模データの場合）
- 入力サイズ $n$、凸包の頂点数 $h$ との関係

**入力パターン**

- 矩形内一様乱数
- 円周上（$h = n$）
- 正規分布
- 格子状

### Qhull との比較

**Python からの Qhull 呼び出し**

```python
from scipy.spatial import ConvexHull
import numpy as np

def convex_hull_qhull(points: List[Point2D]) -> List[Point2D]:
    pts = np.array([(p.x, p.y) for p in points])
    hull = ConvexHull(pts)
    return [Point2D(pts[i, 0], pts[i, 1]) for i in hull.vertices]
```

**比較のポイント**

- 凸包の頂点集合が一致するか（順序は異なってよい）
- 退化ケースで Qhull と同様に動作するか
- 実行時間の差（Qhull は C 実装のため、Python より速いことが多い）

### ベンチマークスクリプトの例

```python
import time
import random

def benchmark(algorithm, points, trials=5):
    times = []
    for _ in range(trials):
        pts = [Point2D(p.x, p.y) for p in points]
        start = time.perf_counter()
        algorithm(pts)
        times.append(time.perf_counter() - start)
    return sum(times) / len(times)

# 使用例
points = random_points_uniform(10000)
t_andrew = benchmark(andrew_monotone_chain, points)
t_qhull = benchmark(convex_hull_qhull, points)
print(f"Andrew: {t_andrew:.4f}s, Qhull: {t_qhull:.4f}s")
```

---

## 20.4 可視化ツールの自作

### 2次元の可視化

```python
import matplotlib.pyplot as plt

def plot_hull_2d(points: List[Point2D], hull: List[Point2D], title="Convex Hull"):
    plt.figure(figsize=(8, 8))
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    plt.scatter(xs, ys, c='blue', alpha=0.5)
    hx = [p.x for p in hull] + [hull[0].x]
    hy = [p.y for p in hull] + [hull[0].y]
    plt.plot(hx, hy, 'r-', linewidth=2)
    plt.title(title)
    plt.axis('equal')
    plt.show()
```

### ステップごとのアニメーション

Graham Scan や Gift Wrapping で、各ステップの凸包候補をフレームとして保存し、`matplotlib.animation` や画像の連続出力でアニメーション化できます。

```python
def animate_graham_scan(points):
    frames = []
    # 各ステップで hull の状態を frames に追加
    ...
    # FuncAnimation で再生
```

### 3次元の可視化

```python
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_hull_3d(faces: List[Tuple[Point3D, Point3D, Point3D]]):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    poly = Poly3DCollection(faces, alpha=0.5, edgecolor='k')
    ax.add_collection3d(poly)
    # 軸の範囲を設定
    ...
    plt.show()
```

インタラクティブな回転が必要な場合は、Plotly や PyVista の利用を検討します。

---

## 20.5 最適化技法

### SIMD 命令の活用

**概要**：複数のデータを同時に演算する SIMD（Single Instruction Multiple Data）により、外積や符号付き体積の一括計算を高速化できます。

**実装**：C/C++ では SSE、AVX、NEON などの intrinsic を直接用いるか、コンパイラの自動ベクトル化に頼ります。Python では `numpy` のベクトル化や `numba` の `@vectorize` で、ある程度の高速化が期待できます。

```python
# numpy によるベクトル化の例
import numpy as np

def cross_batch(O, A, B):
    """複数点の外積を一括計算"""
    return (A[:, 0] - O[0]) * (B[:, 1] - O[1]) - (A[:, 1] - O[1]) * (B[:, 0] - O[0])
```

### マルチスレッド化

**適用場面**

- 複数の独立した凸包計算を並列に実行（例：複数の点群の凸包）
- QuickHull の再帰で、上下凸包や左右の分割を別スレッドで処理
- ベンチマークで複数アルゴリズムを並列実行

**注意**：GIL の影響で、Python の `threading` では CPU バウンドな処理は並列化されにくいです。`multiprocessing` や、C 拡張を使う方が効果的です。

### GPU 活用（参考程度）

凸包の計算を GPU に載せるには、専用のライブラリ（例：CUDA 用の実装）や、GPGPU フレームワークの利用が必要です。大規模な点群では有効な場合がありますが、実装コストは高く、多くの応用では CPU 実装で十分です。

---

## 20.6 ドキュメントとテストの整備

### ドキュメント

- **README**：ライブラリの概要、インストール方法、簡単な使用例
- **API ドキュメント**：関数・クラスの説明、引数、戻り値、例外（Sphinx や pdoc を利用）
- **アルゴリズムの選択ガイド**：第11章の内容をプロジェクト用に要約

### テストの整備

**単体テスト**（第10章を参照）

- 退化ケース（共線、重複、3点以下）
- ランダムデータでの性質チェック（全点が凸包の内側、CCW など）
- Qhull との結果比較

**リグレッションテスト**

- 過去にバグの原因になった入力を `tests/fixtures/` などに保存
- CI で毎回実行し、結果が変わらないことを確認

**テストフレームワーク**

```python
# pytest の例
def test_andrew_vs_qhull():
    for _ in range(100):
        points = random_points_uniform(random.randint(10, 500))
        hull_own = andrew_monotone_chain(points)
        hull_ref = convex_hull_qhull(points)
        assert hulls_equivalent(hull_own, hull_ref, points)
```

---

## 本章のまとめ

- **API 設計**では、複数アルゴリズムを統一インターフェースで提供することを検討する
- **3D 凸包**は、幾何プリミティブ→初期四面体→増分構築の順で段階的に実装する
- **ベンチマーク**で入力パターンを変え、Qhull と比較して正しさと性能を確認する
- **可視化**は理解とデバッグに有効。2D は matplotlib、3D は回転表示を検討する
- **最適化**は、numpy のベクトル化やマルチプロセスから始め、必要に応じて SIMD や GPU を検討する
- **ドキュメントとテスト**を整備し、保守性と信頼性を高める

本教材の第1章から第20章までで、凸包の理論から実装、最適化、周辺の幾何構造までを一通り学びました。ここで得た知識を土台に、より高度な応用や研究に進んでください。
