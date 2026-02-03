"""
Chapter 6: Online Convex Hull animation
- Points arrive one by one; after each addition the convex hull is updated (naive: recompute with Andrew's Monotone Chain).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Tuple, Generator

# ---------------------------------------------------------------------------
# 2D point and geometry
# ---------------------------------------------------------------------------


class Point2D:
    def __init__(self, x: float, y: float):
        self.x, self.y = x, y

    def __eq__(self, other: "Point2D") -> bool:
        return self.x == other.x and self.y == other.y

    def __repr__(self):
        return f"({self.x:.2f}, {self.y:.2f})"


def cross(o: Point2D, a: Point2D, b: Point2D) -> float:
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)


def andrew_monotone_chain(points: List[Point2D]) -> List[Point2D]:
    """Return convex hull (CCW) of points. Used to recompute hull after each addition."""
    n = len(points)
    if n < 3:
        return list(points)
    sorted_pts = sorted(points, key=lambda p: (p.x, p.y))
    lower: List[Point2D] = []
    for p in sorted_pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: List[Point2D] = []
    for p in reversed(sorted_pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


# ---------------------------------------------------------------------------
# Online convex hull: yield state after each addition
# ---------------------------------------------------------------------------


def online_convex_hull_steps(
    points: List[Point2D],
    addition_order: List[int] | None = None,
) -> Generator[Tuple[List[Point2D], List[Point2D], Point2D | None], None, None]:
    """
    Simulate online: add points one by one, recompute hull each time.
    Yield (points_so_far, hull, last_added_point) at each step.
    - points_so_far: points that have arrived so far.
    - hull: convex hull of points_so_far.
    - last_added_point: the point just added (to highlight), or None at step 0.
    """
    n = len(points)
    if n == 0:
        return
    if addition_order is None:
        addition_order = list(np.random.permutation(n))
    # Clamp indices
    addition_order = [i % n for i in addition_order]

    # Step 0: no points yet (optional; we skip and start from 1 point)
    # yield ([], [], None)

    points_so_far: List[Point2D] = []
    for i in range(n):
        idx = addition_order[i]
        p = points[idx]
        points_so_far.append(p)
        hull = andrew_monotone_chain(points_so_far)
        last = p if points_so_far else None
        yield (list(points_so_far), hull, last)


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------


def create_animation(
    points: List[Point2D],
    addition_order: List[int] | None = None,
    interval: int = 120,
    save_path: str | None = None,
):
    """
    Animate online convex hull: points appear one by one, hull updates after each.
    addition_order: indices into points for the order of arrival; None = random.
    """
    steps = list(online_convex_hull_steps(points, addition_order))
    if not steps:
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_aspect("equal")
    ax.set_title("Online Convex Hull — add point, recompute hull", fontsize=14)

    # Use full point set for axis limits so frame is stable
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    margin = 0.1 * (max(max(xs) - min(xs), max(ys) - min(ys)) or 1)
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)

    (scatter_arrived,) = ax.plot([], [], "o", color="lightgray", ms=8, label="Points (arrived)")
    (scatter_new,) = ax.plot(
        [], [], "o", color="orange", ms=12, markeredgewidth=2, markeredgecolor="black", zorder=5, label="Last added"
    )
    (line_hull,) = ax.plot([], [], "b-", lw=2.5, label="Hull")
    text_step = ax.text(0.02, 0.98, "", transform=ax.transAxes, fontsize=11, verticalalignment="top")

    def init():
        scatter_arrived.set_data([], [])
        scatter_new.set_data([], [])
        line_hull.set_data([], [])
        text_step.set_text("")
        return scatter_arrived, scatter_new, line_hull, text_step

    def update(frame_idx):
        if frame_idx >= len(steps):
            return scatter_arrived, scatter_new, line_hull, text_step

        points_so_far, hull, last_added = steps[frame_idx]

        # Points that have arrived so far (gray)
        if points_so_far:
            ax_ = [q.x for q in points_so_far]
            ay_ = [q.y for q in points_so_far]
            scatter_arrived.set_data(ax_, ay_)
        else:
            scatter_arrived.set_data([], [])

        # Last added point (orange)
        if last_added is not None:
            scatter_new.set_data([last_added.x], [last_added.y])
        else:
            scatter_new.set_data([], [])

        # Hull
        if len(hull) >= 2:
            hx = [q.x for q in hull] + [hull[0].x]
            hy = [q.y for q in hull] + [hull[0].y]
            line_hull.set_data(hx, hy)
        elif hull:
            line_hull.set_data([hull[0].x], [hull[0].y])
        else:
            line_hull.set_data([], [])

        step_desc = f"Step {frame_idx + 1} / {len(steps)}"
        step_desc += f"\nPoints: {len(points_so_far)} — hull size: {len(hull)}"
        if last_added is not None:
            step_desc += "\nLast added (orange)"
        text_step.set_text(step_desc)

        return scatter_arrived, scatter_new, line_hull, text_step

    ax.legend(loc="lower left", fontsize=9)
    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(steps) + 8,
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
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)
    n = 18
    points = [Point2D(float(x), float(y)) for x, y in np.random.rand(n, 2) * 8 + 1]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    gif_path = os.path.join(script_dir, "online_convex_hull.gif")
    print("Starting Online Convex Hull animation (close window to exit)")
    create_animation(points, addition_order=None, interval=180, save_path=gif_path)
