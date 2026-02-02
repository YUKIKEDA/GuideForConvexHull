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

```
GiftWrapping3D(P):
    if |P| < 4:
        return 適切に処理（0〜3点、共面など）

    // 初期面を探す：3点が同一直線上でないもの
    (a, b, c) = 初期面を構成する3点
    // 法線が外側を向くように (a, b, c) の順序を調整
    faces = {(a, b, c)}
    horizon = {(a,b), (b,c), (c,a)}  // 稜線（向き付き）

    while horizon が空でない:
        (a, b) = horizon から1本取り出す
        p = None
        for each q in P:
            if q は a, b と重複または共線: continue
            if すべての r in P に対して signed_volume(a, b, q, r) <= 0:
                if p is None or tie_break(q が p より適切:
                    p = q
        if p is None: continue  // 退化ケース
        faces.add((a, b, p))
        // 新たな稜線 (b, p), (p, a) を horizon に追加
        // (a, b) の逆向きがすでに処理済みなら horizon から削除
        horizon を更新

    return faces
```

### Python 実装の骨格

```python
from collections import deque

def gift_wrapping_3d(points):
    n = len(points)
    if n < 4:
        return []  # または点数に応じた処理

    pts = [tuple(p) for p in points]  # (x, y, z)

    def signed_volume(a, b, c, d):
        ab = (b[0]-a[0], b[1]-a[1], b[2]-a[2])
        ac = (c[0]-a[0], c[1]-a[1], c[2]-a[2])
        ad = (d[0]-a[0], d[1]-a[1], d[2]-a[2])
        cx = ab[1]*ac[2] - ab[2]*ac[1]
        cy = ab[2]*ac[0] - ab[0]*ac[2]
        cz = ab[0]*ac[1] - ab[1]*ac[0]
        return cx*ad[0] + cy*ad[1] + cz*ad[2]

    # 初期面を探す
    a, b, c = None, None, None
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if signed_volume(pts[i], pts[j], pts[k], pts[(k+1) % n if k+1 < n else 0]) != 0:
                    a, b, c = pts[i], pts[j], pts[k]
                    break
            if a is not None: break
        if a is not None: break

    if a is None:
        return []  # 全点が共面

    # 法線が外側を向くようにする（1点を反対側に）
    for i in range(n):
        if pts[i] not in (a, b, c) and signed_volume(a, b, c, pts[i]) > 0:
            b, c = c, b
            break

    faces = set()
    faces.add((a, b, c))
    horizon = deque([(a, b), (b, c), (c, a)])

    while horizon:
        a, b = horizon.popleft()
        best = None
        for q in pts:
            if q in (a, b): continue
            vol = signed_volume(a, b, q, pts[0] if pts[0] not in (a,b,q) else pts[1])
            if vol > 0: continue
            ok = all(signed_volume(a, b, q, r) <= 0 for r in pts if r not in (a, b, q))
            if ok and (best is None or signed_volume(a, b, best, q) < 0):
                best = q
        if best is None: continue
        p = best
        faces.add((a, b, p))
        if (b, p) not in [(f[0],f[1]) for f in faces] + [(f[1],f[2]) for f in faces] + [(f[2],f[0]) for f in faces]:
            horizon.append((b, p))
        if (p, a) not in ...:
            horizon.append((p, a))

    return list(faces)
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
