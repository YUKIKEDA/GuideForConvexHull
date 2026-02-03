"""
Chapter 8: Chan's Algorithm animation
- Split into groups of size m, compute each group's hull with Graham Scan,
  then Gift Wrapping over the small hulls (tangent point per hull, pick best).
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


def ccw(o: Point2D, a: Point2D, b: Point2D) -> int:
    c = cross(o, a, b)
    if c > 0:
        return 1
    if c < 0:
        return -1
    return 0


def dist_sq(a: Point2D, b: Point2D) -> float:
    dx = b.x - a.x
    dy = b.y - a.y
    return dx * dx + dy * dy


# ---------------------------------------------------------------------------
# Graham Scan (Andrew's Monotone Chain) for small hulls
# ---------------------------------------------------------------------------


def graham_scan(points: List[Point2D]) -> List[Point2D]:
    """Return convex hull (CCW) of points."""
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
# Right tangent: point p to convex polygon hull (CCW) -> tangent point index
# ---------------------------------------------------------------------------


def right_tangent(p: Point2D, hull: List[Point2D]) -> int:
    """Index of tangent point: from p, the vertex that minimizes polar angle (right tangent)."""
    n = len(hull)
    if n <= 1:
        return 0
    best = 0
    best_angle = np.arctan2(hull[0].y - p.y, hull[0].x - p.x)
    for i in range(1, n):
        angle = np.arctan2(hull[i].y - p.y, hull[i].x - p.x)
        if angle < best_angle:
            best_angle = angle
            best = i
    return best


# ---------------------------------------------------------------------------
# Chan's Algorithm: yield state for animation
# ---------------------------------------------------------------------------

# State: (phase, hulls, current, tangent_points, main_hull)
# phase: "hulls" | "jarvis" | "done"
# hulls: list of small hulls (each list of points)
# current: current vertex (Jarvis step)
# tangent_points: list of tangent points (one per small hull) when phase=="jarvis"
# main_hull: main convex hull built so far


def chan_steps(
    points: List[Point2D],
    m: int = 8,
) -> Generator[Tuple[str, List[List[Point2D]], Point2D | None, List[Point2D], List[Point2D]], None, None]:
    """
    Yield (phase, hulls, current, tangent_points, main_hull) at each step.
    Uses fixed m (no retry loop) so animation is deterministic; m should be >= h.
    """
    n = len(points)
    if n == 0:
        return
    if n < 3:
        yield ("done", [], None, [], list(points))
        return

    m = min(m, n)
    if m < 2:
        yield ("done", [], None, [], list(points))
        return

    # 1. Split into groups and compute small hulls
    hulls: List[List[Point2D]] = []
    for i in range(0, n, m):
        group = points[i : i + m]
        if len(group) >= 3:
            hulls.append(graham_scan(group))
        elif len(group) == 2:
            hulls.append(list(group))
        elif len(group) == 1:
            hulls.append(list(group))

    yield ("hulls", [list(h) for h in hulls], None, [], [])

    # 2. Gift Wrapping on hulls: start from leftmost
    start = min(points, key=lambda p: (p.x, p.y))
    main_hull: List[Point2D] = [start]
    current = start

    for _ in range(m):
        next_pt = None
        tangent_pts = []
        for h in hulls:
            if not h:
                continue
            idx = right_tangent(current, h)
            q = h[idx]
            tangent_pts.append(q)
            if next_pt is None:
                next_pt = q
                continue
            c = ccw(current, next_pt, q)
            if c > 0:
                next_pt = q
            elif c == 0 and dist_sq(current, q) > dist_sq(current, next_pt):
                next_pt = q

        yield ("jarvis", [list(h) for h in hulls], current, list(tangent_pts), list(main_hull))

        if next_pt is None or next_pt == start:
            break
        main_hull.append(next_pt)
        current = next_pt

    yield ("done", [list(h) for h in hulls], None, [], list(main_hull))


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------


def create_animation(
    points: List[Point2D],
    m: int = 8,
    interval: int = 120,
    save_path: str | None = None,
):
    steps = list(chan_steps(points, m=m))
    if not steps:
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_aspect("equal")
    ax.set_title("Chan's Algorithm — O(n log h) Convex Hull", fontsize=14)

    xs = [p.x for p in points]
    ys = [p.y for p in points]
    margin = 0.1 * (max(max(xs) - min(xs), max(ys) - min(ys)) or 1)
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)

    (scatter_all,) = ax.plot([], [], "o", color="lightgray", ms=6, label="Points")
    (scatter_current,) = ax.plot(
        [], [], "o", color="green", ms=14, markeredgewidth=2, markeredgecolor="black", zorder=6, label="Current"
    )
    (scatter_tangent,) = ax.plot(
        [], [], "o", color="orange", ms=10, markeredgewidth=2, markeredgecolor="black", zorder=5, label="Tangent"
    )
    (line_hull,) = ax.plot([], [], "b-", lw=2.5, label="Main hull")
    line_small_hulls: List[Tuple] = []
    text_step = ax.text(0.02, 0.98, "", transform=ax.transAxes, fontsize=11, verticalalignment="top")

    # Pre-create lines for small hulls (we'll reuse a few)
    colors = ["gray", "silver", "darkgray", "lightgray"]
    for _ in range(12):
        (ln,) = ax.plot([], [], "-", lw=1.2, alpha=0.7, color=colors[_ % len(colors)])
        line_small_hulls.append(ln)

    def init():
        scatter_all.set_data([], [])
        scatter_current.set_data([], [])
        scatter_tangent.set_data([], [])
        line_hull.set_data([], [])
        for ln in line_small_hulls:
            ln.set_data([], [])
        text_step.set_text("")
        return [scatter_all, scatter_current, scatter_tangent, line_hull, text_step] + line_small_hulls

    def update(frame_idx):
        if frame_idx >= len(steps):
            return [scatter_all, scatter_current, scatter_tangent, line_hull, text_step] + line_small_hulls

        phase, hulls, current, tangent_points, main_hull = steps[frame_idx]

        scatter_all.set_data(xs, ys)

        for i, ln in enumerate(line_small_hulls):
            if i < len(hulls) and len(hulls[i]) >= 2:
                h = hulls[i]
                hx = [q.x for q in h] + [h[0].x]
                hy = [q.y for q in h] + [h[0].y]
                ln.set_data(hx, hy)
                ln.set_visible(True)
            else:
                ln.set_visible(False)

        if current is not None:
            scatter_current.set_data([current.x], [current.y])
        else:
            scatter_current.set_data([], [])

        if tangent_points:
            tx = [q.x for q in tangent_points]
            ty = [q.y for q in tangent_points]
            scatter_tangent.set_data(tx, ty)
        else:
            scatter_tangent.set_data([], [])

        if len(main_hull) >= 2:
            mx = [q.x for q in main_hull] + [main_hull[0].x]
            my_ = [q.y for q in main_hull] + [main_hull[0].y]
            line_hull.set_data(mx, my_)
        else:
            line_hull.set_data([], [])

        step_desc = f"Step {frame_idx + 1} / {len(steps)}"
        if phase == "hulls":
            step_desc += "\nSmall hulls (Graham Scan per group)"
        elif phase == "jarvis":
            step_desc += "\nGift Wrapping: tangent points (orange), pick best"
        else:
            step_desc += "\nHull complete"
        text_step.set_text(step_desc)

        return [scatter_all, scatter_current, scatter_tangent, line_hull, text_step] + line_small_hulls

    ax.legend(loc="lower left", fontsize=9)
    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(steps) + 6,
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
    n = 24
    points = [Point2D(float(x), float(y)) for x, y in np.random.rand(n, 2) * 8 + 1]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    gif_path = os.path.join(script_dir, "chan_algorithm.gif")
    print("Starting Chan's Algorithm animation (close window to exit)")
    create_animation(points, m=8, interval=180, save_path=gif_path)
