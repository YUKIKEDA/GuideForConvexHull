"""
第3章 Gift Wrapping（Jarvis March）のアニメーション
- 最も左の点から始め、反時計回りに「最も曲がった点」を順に選んで凸包を構築する様子を可視化
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Tuple, Generator

# ---------------------------------------------------------------------------
# 2次元点と幾何関数（第1章・第2章に相当）
# ---------------------------------------------------------------------------


class Point2D:
    def __init__(self, x: float, y: float):
        self.x, self.y = x, y

    def __eq__(self, other: "Point2D") -> bool:
        return self.x == other.x and self.y == other.y

    def __repr__(self):
        return f"({self.x:.2f}, {self.y:.2f})"


def cross(o: Point2D, a: Point2D, b: Point2D) -> float:
    """外積: >0 左折, =0 同一直線, <0 右折"""
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)


def ccw(o: Point2D, a: Point2D, b: Point2D) -> int:
    """反時計回りなら 1, 同一直線上なら 0, 時計回りなら -1"""
    c = cross(o, a, b)
    if c > 0:
        return 1
    if c < 0:
        return -1
    return 0


def dist_sq(a: Point2D, b: Point2D) -> float:
    """距離の2乗（比較用）"""
    dx = b.x - a.x
    dy = b.y - a.y
    return dx * dx + dy * dy


# ---------------------------------------------------------------------------
# Gift Wrapping：アニメーション用に状態を yield する版
# ---------------------------------------------------------------------------


def gift_wrapping_steps(
    points: List[Point2D],
) -> Generator[Tuple[List[Point2D], Point2D, Point2D | None], None, None]:
    """
    Gift Wrapping の各ステップを yield する。
    各 yield: (hull_so_far, current, candidate)
    - hull_so_far: ここまで確定した凸包の頂点列
    - current: 現在の頂点（ここから「次の頂点」を探している）
    - candidate: 今比較している候補点（None のときは比較中でない）
    """
    n = len(points)
    if n == 0:
        return
    if n < 3:
        yield (list(points), points[0], None)
        return

    start = min(points, key=lambda p: (p.x, p.y))
    hull = [start]
    current = start

    # 開始直後: スタート点だけの状態
    yield (list(hull), current, None)

    while True:
        next_pt = None
        # 候補を1つずつ比較するたびに yield（アニメーション用）
        for p in points:
            if p == current:
                continue
            if next_pt is None:
                next_pt = p
                yield (list(hull), current, next_pt)
                continue
            yield (list(hull), current, p)
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
        yield (list(hull), current, None)

    # 最後は hull 全体（閉じた凸包）を表示するため、start を再度 yield
    yield (hull[:-1], None, None)


# ---------------------------------------------------------------------------
# アニメーション
# ---------------------------------------------------------------------------


def create_animation(
    points: List[Point2D],
    interval: int = 120,
    save_path: str | None = None,
):
    """
    Gift Wrapping のアニメーションを作成する。
    - points: 入力点集合
    - interval: フレーム間隔（ミリ秒）
    - save_path: 指定すると GIF を保存（例: "gift_wrapping.gif"）
    """
    steps = list(gift_wrapping_steps(points))
    if not steps:
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_aspect("equal")
    ax.set_title("Gift Wrapping (Jarvis March) — 2D Convex Hull", fontsize=14)

    # 全点の座標
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    margin = 0.1 * (max(max(xs) - min(xs), max(ys) - min(ys)) or 1)
    x_min, x_max = min(xs) - margin, max(xs) + margin
    y_min, y_max = min(ys) - margin, max(ys) + margin
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # 描画オブジェクト
    (scatter_all,) = ax.plot([], [], "o", color="lightgray", ms=8, label="Points")
    (scatter_current,) = ax.plot(
        [], [], "o", color="darkgreen", ms=14, markeredgewidth=2, markeredgecolor="black", zorder=5, label="Current vertex"
    )
    (scatter_candidate,) = ax.plot(
        [], [], "o", color="orange", ms=12, markeredgewidth=2, markeredgecolor="black", zorder=5, label="Candidate"
    )
    (line_hull,) = ax.plot([], [], "b-", lw=2.5, label="Hull (fixed)")
    (line_search,) = ax.plot([], [], "m--", lw=1.5, alpha=0.8, label="Line to candidate")
    text_step = ax.text(0.02, 0.98, "", transform=ax.transAxes, fontsize=11, verticalalignment="top")

    def init():
        scatter_all.set_data([], [])
        scatter_current.set_data([], [])
        scatter_candidate.set_data([], [])
        line_hull.set_data([], [])
        line_search.set_data([], [])
        text_step.set_text("")
        return scatter_all, scatter_current, scatter_candidate, line_hull, line_search, text_step

    def update(frame_idx):
        if frame_idx >= len(steps):
            return scatter_all, scatter_current, scatter_candidate, line_hull, line_search, text_step

        hull_so_far, current, candidate = steps[frame_idx]

        # 全点
        scatter_all.set_data(xs, ys)

        # 確定した凸包の辺
        if len(hull_so_far) >= 2:
            hx = [p.x for p in hull_so_far] + [hull_so_far[0].x]
            hy = [p.y for p in hull_so_far] + [hull_so_far[0].y]
            line_hull.set_data(hx, hy)
        else:
            line_hull.set_data([], [])

        # 現在の頂点
        if current is not None:
            scatter_current.set_data([current.x], [current.y])
        else:
            scatter_current.set_data([], [])

        # 比較中の候補と、current → candidate の線
        if candidate is not None and current is not None:
            scatter_candidate.set_data([candidate.x], [candidate.y])
            line_search.set_data([current.x, candidate.x], [current.y, candidate.y])
        else:
            scatter_candidate.set_data([], [])
            line_search.set_data([], [])

        # Step description
        step_desc = f"Step {frame_idx + 1} / {len(steps)}"
        if current is not None and candidate is not None:
            step_desc += "\nComparing candidate"
        elif current is not None and candidate is None:
            step_desc += "\nFinding next vertex"
        else:
            step_desc += "\nHull complete"
        text_step.set_text(step_desc)

        return scatter_all, scatter_current, scatter_candidate, line_hull, line_search, text_step

    ax.legend(loc="lower left", fontsize=9)
    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(steps) + 5,  # 最後に少し止める
        interval=interval,
        blit=False,
        repeat=True,
    )

    plt.tight_layout()
    if save_path:
        anim.save(save_path, writer="pillow", fps=1000 // interval)
        plt.close(fig)
    else:
        plt.show()
    return anim


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    np.random.seed(42)
    n = 20
    points = [Point2D(float(x), float(y)) for x, y in np.random.rand(n, 2) * 8 + 1]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    gif_path = os.path.join(script_dir, "gift_wrapping.gif")
    print("Starting Gift Wrapping animation (close window to exit)")
    create_animation(points, interval=150, save_path=gif_path)
