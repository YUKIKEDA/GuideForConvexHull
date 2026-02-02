# 付録D 参考文献・リソース

本教材の学習を深めるための教科書、ドキュメント、オンラインリソースをまとめます。

---

## 定番教科書

### 英語

| 書名 | 著者 | 出版社 | 備考 |
|------|------|--------|------|
| **Computational Geometry: Algorithms and Applications** | Mark de Berg, Marc van Kreveld, Mark Overmars, Otfried Schwarzkopf | Springer | 第3版（2008年）。凸包・ボロノイ・ドロネー等を系統的に扱う定番教科書。擬似コードと図が豊富。 |
| **Computational Geometry in C** | Joseph O'Rourke | Cambridge University Press | 2版（1998年）。C による実装が中心。凸包の Gift Wrapping、Graham Scan など。 |
| **Computational Geometry: An Introduction** | Franco P. Preparata, Michael Ian Shamos | Springer | 1985年。計算幾何の古典。理論的な基礎に強い。 |
| **Algorithmic Geometry** | Jean-Daniel Boissonnat, Mariette Yvinec, François Cazals | Cambridge University Press | CGAL の設計思想に近い。厳密な数値計算にも触れる。 |

### 日本語

| 書名 | 著者 | 出版社 | 備考 |
|------|------|--------|------|
| **計算幾何学：アルゴリズムと応用** | M. de Berg 他（浅野哲夫訳） | 近代科学社 | 上記 de Berg の日本語訳。 |
| **プログラミングコンテストチャレンジブック** | 秋葉拓哉・岩田陽一・北川宜稔 | マイナビ出版 | 競技プログラミング向け。凸包や幾何の基本を簡潔に解説。 |

---

## Qhull 公式ドキュメント

| リソース | URL | 内容 |
|----------|-----|------|
| **Qhull ホーム** | https://www.qhull.org/ | 概要、ダウンロード、ニュース |
| **マニュアル** | https://www.qhull.org/html/index.htm | Qhull と rbox の詳細マニュアル |
| **プログラム・オプション** | https://www.qhull.org/html/qh-quick.htm | qconvex, qdelaunay 等のクイックリファレンス |
| **FAQ** | https://www.qhull.org/html/qh-faq.htm | よくある質問 |
| **qconvex** | https://www.qhull.org/html/qconvex.htm | 凸包用コマンドの説明 |
| **数値精度（Imprecision）** | https://www.qhull.org/html/qh-impre.htm | 浮動小数点誤差とオプション（QJ 等） |
| **GitHub** | https://github.com/qhull/qhull | ソースコード、C++ インターフェース情報 |
| **ダウンロード** | https://www.qhull.org/download/ | ソース・バイナリ |

### 原論文

- Barber, C.B., Dobkin, D.P., Huhdanpaa, H.T., **"The Quickhull algorithm for convex hulls,"** *ACM Trans. on Mathematical Software*, 22(4):469–483, Dec 1996.  
  Qhull の基となる QuickHull の論文。ACM Digital Library や CiteSeer で検索可能。

---

## オンラインリソース

### 計算幾何全般

| サイト | URL | 内容 |
|--------|-----|------|
| **Jeff Erickson's Computational Geometry** | https://jeffe.cs.illinois.edu/compgeom/ | 講義ノート、ソフトウェア一覧、教科書リンク集 |
| **computational-geometry.org** | http://www.computational-geometry.org/ | 学会・会議・論文へのリンク |
| **CGAL** | https://www.cgal.org/ | C++ 計算幾何ライブラリ。凸包、ボロノイ、ドロネー等 |
| **Geometry Center Software** | https://www.geom.uiuc.edu/software/cglist | 計算幾何ソフトウェア一覧（Amenta 監修） |

### 凸包・アルゴリズム解説

| サイト | URL | 内容 |
|--------|-----|------|
| **CP-Algorithms (Convex Hull)** | https://cp-algorithms.com/geometry/convex-hull.html | Graham Scan, Andrew's 等の簡潔な実装 |
| **Wikipedia: Convex Hull** | https://en.wikipedia.org/wiki/Convex_hull | 定義、アルゴリズム一覧、応用 |
| **Wikipedia: Chan's Algorithm** | https://en.wikipedia.org/wiki/Chan%27s_algorithm | O(n log h) アルゴリズムの解説 |
| **Stony Brook Algorithm Repository** | http://www.cs.sunysb.edu/~algorith/major_section/1.6.shtml | 計算幾何アルゴリズムの概要 |

### ライブラリ・ツール

| ツール | URL | 備考 |
|--------|-----|------|
| **SciPy spatial** | https://docs.scipy.org/doc/scipy/reference/spatial.html | `ConvexHull`, `Delaunay` 等。内部で Qhull を使用 |
| **MATLAB convhull** | https://www.mathworks.com/help/matlab/ref/convhull.html | n 次元凸包。Qhull ベース |
| **R geometry** | https://CRAN.R-project.org/package=geometry | Qhull を R から利用 |
| **Geomview** | http://www.geomview.org/ | Qhull 出力の 3D・4D 可視化 |

---

## 学習の進め方

1. **理論の深化**：de Berg の教科書で凸包の数学的定義とアルゴリズムの正当性を理解する。
2. **実装の確認**：Qhull のマニュアルでオプション（QJ, Qt 等）や数値精度の扱いを確認する。
3. **比較検証**：SciPy の `ConvexHull` で自作実装の結果を検証する。
4. **発展**：CGAL や Jeff Erickson の講義ノートで高次元・応用に触れる。
