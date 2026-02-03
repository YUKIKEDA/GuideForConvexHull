# 第13章 3次元 Gift Wrapping

2次元の Gift Wrapping は「最も左の点から始めて、反時計回りに最も曲がった点を選んでいく」アルゴリズムでした。3次元では、**稜線（edge）**を単位に、凸包の**面**を1つずつ追加していく形に拡張されます。この章では、3次元 Gift Wrapping（Jarvis March の3次元版）の考え方と実装の概要を扱います。

---

## 13.1 アイデアの 2D からの拡張

### 2次元の復習

2次元 Gift Wrapping では：

1. 凸包の頂点を1つ決める（例：最も左の点）
2. その頂点から「反時計回りに最も曲がった点」を次の頂点とする
3. スタートに戻るまで繰り返す

各ステップでは**頂点**を単位に、凸包の**辺**を1本ずつ決めていました。

### 3次元への拡張

3次元では：

1. 凸包の**稜線（辺）** を1本選ぶ
2. その稜線を共有する**次の面**を決める（稜線の向きに対して「最も外側に曲がった」点を選ぶ）
3. 新しい面が決まると新しい稜線が現れる。未処理の稜線について同様の操作を繰り返す
4. すべての稜線を処理し終えたら完了

つまり、**稜線**を単位に、凸包の**面**を1つずつ見つけていきます。2次元の「頂点→次の頂点」が、3次元では「稜線→次の面（＝次の稜線群）」に対応します。

---

## 13.2 稜線ベースのアルゴリズム

### 用語

- **稜線（edge）**：凸包の2頂点を結ぶ辺。ちょうど2つの面に属する
- **ホライゾン（horizon）**：現時点で「外側に開いている」稜線の集合。ギフト包装で言えば、まだ包装されていない縁の部分
- **候補面**：稜線 $(a, b)$ と候補点 $p$ で決まる三角形 $(a, b, p)$

### 稜線から次の面を決める

稜線 $(a, b)$ が与えられたとき、凸包の次の面は、$(a, b)$ を含む三角形 $(a, b, p)$ の形をしています。$p$ は、次の条件を満たす点です。

- 平面 $(a, b, p)$ に対して、**すべての点が同じ側**にある（凸包の外側から見て、他の点は面の「内側」にある）

言い換えると、$p$ は「稜線 $(a, b)$ の周りを回ったとき、最初に当たる」点です。これを探すために、全点 $p$ について、$\text{signed\_volume}(a, b, p, q)$ の符号を全点 $q$ に対して確認します。

**判定**：法線の向きを「凸包の外側」に合わせておくとき、すべての $q$ に対して $\text{signed\_volume}(a, b, p, q) \leq 0$ となる $p$ が、次の面の第3頂点です。複数ある場合は、同一平面上の退化ケースとして、適切な tie-break（距離など）で1つに決めます。

### アルゴリズムの流れ

```
1. 初期面を1つ作る
   - 3点が同一直線上にない最初の面 (p0, p1, p2) を選ぶ
   - 法線の向きを「外側」に統一する（全点が負の側にあるようにする）

2. 未処理の稜線の集合（ホライゾン）を管理する
   - 初期面の3辺をホライゾンに加える

3. ホライゾンが空でない間：
   - 稜線 (a, b) を1本取り出す
   - 全点 p を調べ、平面 (a, b, p) に対して全点が同一側にある p を求める
   - その p を使って新面 (a, b, p) を追加
   - 新面のうち、まだ処理していない稜線をホライゾンに加える
   - すでに両側が面で埋まっている稜線はホライゾンから除く
```

以下のアニメーションは、3次元 Gift Wrapping の動作を可視化したものです（`Chapter13/animation.py` で生成）。稜線を1本取り出し、候補点を調べて次の面を追加する流れを示しています。

![3次元 Gift Wrapping アニメーション](gift_wrapping_3d.gif)

---

## 13.3 計算量：O(nh)

- $n$：点数
- $h$：凸包の面数（オイラー公式より $h$ は頂点数・辺数と同程度）

**各面を決めるステップ**：1本の稜線について、全 $n$ 点を候補とし、それぞれについて全点との符号付き体積を調べる → $O(n^2)$  per edge... 実際には、稜線ごとに「次の点」を探すのに $O(n)$ かかります。

**面数**：凸包の面数は $O(h)$（頂点数を $h'$ とすると、三角形メッシュで $F = 2h' - 4$ 程度）。

**合計**：各稜線（$O(h)$ 本）について $O(n)$ で次の点を探すと、全体で **$O(nh)$** になります。2次元 Gift Wrapping と同じく、出力サイズ $h$ に比例します。

---

## 13.4 実装

### 疑似コード

以下は、制御構造と代入をアルゴリズムの教科書でよく使う記法で書いた疑似コードである。代入は `←`、等号は `=`、ベクトルの減算・外積・内積はそれぞれ `−`, `×`, `·` で表す。

```
SIGNED-VOLUME(a, b, c, d)
    ab ← b − a
    ac ← c − a
    ad ← d − a
    return (ab × ac) · ad
```

```
MAKE-INITIAL-FACE(P)
    for i ← 1 to |P| − 2 do
        for j ← i + 1 to |P| − 1 do
            for k ← j + 1 to |P| do
                p0 ← P[i], p1 ← P[j], p2 ← P[k]
                if (p1 − p0) × (p2 − p0) ≠ 0 then
                    flip ← false
                    for each q in P, q ∉ {p0, p1, p2} do
                        if SIGNED-VOLUME(p0, p1, p2, q) > 0 then flip ← true; break
                    if flip then return (p0, p2, p1)   // flip normal outward
                    else return (p0, p1, p2)
    return NIL
```

```
FIND-NEXT-FACE(P, a, b)
    for each p in P such that p ≠ a and p ≠ b do
        valid ← true
        for each q in P such that q ∉ {a, b, p} do
            if SIGNED-VOLUME(a, b, p, q) > 0 then valid ← false; break
        if valid then
            // if multiple candidates, tie-break (e.g. by distance from a)
            return p
    return NIL
```

```
GIFT-WRAPPING-3D(P)
    if |P| < 4 then return NIL
    face0 ← MAKE-INITIAL-FACE(P)
    if face0 = NIL then return NIL
    faces ← { face0 }
    horizon ← ∅
    (p0, p1, p2) ← face0
    for e ∈ { {p0,p1}, {p1,p2}, {p2,p0} } do
        edge_face_count[e] ← 1
        horizon ← horizon ∪ { e }

    while horizon ≠ ∅ do
        e ← an arbitrary element of horizon
        horizon ← horizon \ { e }
        a, b ← the two vertices of e
        p ← FIND-NEXT-FACE(P, a, b)
        if p = NIL then continue
        f ← (a, b, p)
        faces ← faces ∪ { f }
        for e'' ∈ { {a,b}, {b,p}, {p,a} } do
            edge_face_count[e''] ← edge_face_count[e''] + 1
        for e' ∈ { {a,p}, {p,b} } do   // the two edges of f other than e = {a,b}
            if edge_face_count[e'] = 1 then horizon ← horizon ∪ { e' }
            else horizon ← horizon \ { e' }
    return faces
```

### Python 実装の骨格

```python
def signed_volume(a, b, c, d):
    """四面体 (a,b,c,d) の符号付き体積の6倍。法線 (a,b,c) の正の側に d があると正。"""
    ab = (b[0]-a[0], b[1]-a[1], b[2]-a[2])
    ac = (c[0]-a[0], c[1]-a[1], c[2]-a[2])
    ad = (d[0]-a[0], d[1]-a[1], d[2]-a[2])
    return dot3d(cross3d(ab, ac), ad)

def make_initial_face(points):
    """同一直線上にない3点で初期面 (i, j, k) を返す。法線は外側に。"""
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a, b, c = points[i], points[j], points[k]
                if cross_norm_sq(a, b, c) < eps: continue  # 共線
                # 他全点が負の側になるよう法線を向ける（必要なら (i,k,j) を返す）
                ...
    return None

def find_next_face(points, ai, bi):
    """稜線 (points[ai], points[bi]) に対する次の面の第3頂点のインデックス。"""
    a, b = points[ai], points[bi]
    for p_idx in range(len(points)):
        if p_idx in (ai, bi): continue
        p = points[p_idx]
        # すべての q（q ≠ a, b, p）について signed_volume(a, b, p, q) ≤ 0 か確認
        if all(signed_volume(a, b, p, points[q_idx]) <= 0
               for q_idx in range(len(points)) if q_idx not in (ai, bi, p_idx)):
            # 複数候補なら tie-break（距離など）で1つに
            return p_idx
    return None

def gift_wrapping_3d(points):
    if len(points) < 4: return []
    face0 = make_initial_face(points)
    if face0 is None: return []
    faces = [face0]
    edge_face_count = {}  # 稜線 frozenset({i,j}) → 属する面の数
    i0, j0, k0 = face0
    for e in [frozenset({i0,j0}), frozenset({j0,k0}), frozenset({k0,i0})]:
        edge_face_count[e] = 1
    horizon = set(edge_face_count.keys())

    while horizon:
        e = next(iter(horizon))
        ai, bi = tuple(e)
        p_idx = find_next_face(points, ai, bi)
        if p_idx is None:
            horizon.discard(e)
            continue
        new_face = (ai, bi, p_idx)
        faces.append(new_face)
        # 新面の3辺を edge_face_count に反映
        for (a, b, c) in [new_face]:
            for e2 in [frozenset({a,b}), frozenset({b,c}), frozenset({c,a})]:
                edge_face_count[e2] = edge_face_count.get(e2, 0) + 1
        horizon.discard(e)  # 稜線 e は2面に属するので horizon から除く
        # 新面の残り2辺 (ai,p_idx), (p_idx,bi): 1面のみなら horizon に追加、2面なら除く
        for e2 in [frozenset({ai, p_idx}), frozenset({p_idx, bi})]:
            if edge_face_count[e2] == 1:
                horizon.add(e2)
            else:
                horizon.discard(e2)
    return faces
```

> 注：上記は概念を示すための骨格です。稜線の重複管理、退化ケース、法線の一貫性などは、実装時にきちんと詰める必要があります。実用には、第15章の Half-edge や DCEL を用いた実装が扱いやすいです。

### 実装の注意点

- **初期面**：全点が同一平面上のときは凸包は2次元的になるため、別処理が必要
- **稜線の向き**：同じ辺を (a,b) と (b,a) で重複して扱わないように、稜線の向きを統一する
- **退化ケース**：4点が同一平面上にある場合、面の選び方で tie-break が必要

---

## 本章のまとめ

- **3次元 Gift Wrapping** は、稜線を単位に凸包の面を1つずつ決めていくアルゴリズムである
- 各稜線 $(a, b)$ について、平面 $(a, b, p)$ の「外側」に他の点が来ないような $p$ を探す
- 計算量は $O(nh)$ で、2次元と同様の構造を持つ
- 稜線の管理や退化ケースの扱いがやや複雑なため、第15章のデータ構造を併用すると実装しやすくなる

次章では、**3次元 QuickHull** と**増分構築法**を扱い、より実用的な3次元凸包アルゴリズムを学びます。
