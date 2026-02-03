"""
Chapter 4: Graham Scan animation
- Polar-angle sort around base (y-min), then build hull with stack (pop on right turn, push on left).
"""

import math
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Tuple, Generator

# ---------------------------------------------------------------------------
# 2D point and geometry (same as Chapter 2/3)
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
# Graham Scan: yield state for animation (stack, current_point)
# ---------------------------------------------------------------------------


def graham_scan_steps(
    points: List[Point2D],
) -> Generator[Tuple[List[Point2D], Point2D | None], None, None]:
    """
    Yield (stack, current_point) at each step.
    - stack: current hull vertex stack (bottom to top).
    - current_point: point we are considering to add (None when idle or after push).
    """
    n = len(points)
    if n == 0:
        return
    if n < 3:
        yield (list(points), None)
        return

    base = min(points, key=lambda p: (p.y, p.x))
    others = [p for p in points if p != base]

    def sort_key(p: Point2D) -> Tuple[float, float]:
        angle = math.atan2(p.y - base.y, p.x - base.x)
        return (angle, dist_sq(base, p))

    others.sort(key=sort_key)
    stack: List[Point2D] = [base]

    yield (list(stack), None)

    for p in others:
        yield (list(stack), p)
        while len(stack) >= 2 and ccw(stack[-2], stack[-1], p) <= 0:
            stack.pop()
            yield (list(stack), p)
        stack.append(p)
        yield (list(stack), None)

    yield (list(stack), None)


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------


def create_animation(
    points: List[Point2D],
    interval: int = 120,
    save_path: str | None = None,
):
    steps = list(graham_scan_steps(points))
    if not steps:
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_aspect("equal")
    ax.set_title("Graham Scan — 2D Convex Hull", fontsize=14)

    xs = [p.x for p in points]
    ys = [p.y for p in points]
    margin = 0.1 * (max(max(xs) - min(xs), max(ys) - min(ys)) or 1)
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)

    (scatter_all,) = ax.plot([], [], "o", color="lightgray", ms=8, label="Points")
    (scatter_base,) = ax.plot(
        [], [], "o", color="green", ms=12, markeredgewidth=2, markeredgecolor="black", zorder=5, label="Base"
    )
    (scatter_current,) = ax.plot(
        [], [], "o", color="orange", ms=12, markeredgewidth=2, markeredgecolor="black", zorder=5, label="Current"
    )
    (line_hull,) = ax.plot([], [], "b-", lw=2.5, label="Hull (stack)")
    (line_candidate,) = ax.plot([], [], "m--", lw=1.5, alpha=0.8, label="To current")
    text_step = ax.text(0.02, 0.98, "", transform=ax.transAxes, fontsize=11, verticalalignment="top")

    def init():
        scatter_all.set_data([], [])
        scatter_base.set_data([], [])
        scatter_current.set_data([], [])
        line_hull.set_data([], [])
        line_candidate.set_data([], [])
        text_step.set_text("")
        return scatter_all, scatter_base, scatter_current, line_hull, line_candidate, text_step

    def update(frame_idx):
        if frame_idx >= len(steps):
            return scatter_all, scatter_base, scatter_current, line_hull, line_candidate, text_step

        stack, current = steps[frame_idx]

        scatter_all.set_data(xs, ys)

        if stack:
            base_pt = stack[0]
            scatter_base.set_data([base_pt.x], [base_pt.y])
            if len(stack) >= 2:
                hx = [q.x for q in stack] + [stack[0].x]
                hy = [q.y for q in stack] + [stack[0].y]
                line_hull.set_data(hx, hy)
            else:
                line_hull.set_data([], [])
        else:
            scatter_base.set_data([], [])
            line_hull.set_data([], [])

        if current is not None:
            scatter_current.set_data([current.x], [current.y])
            if stack:
                line_candidate.set_data([stack[-1].x, current.x], [stack[-1].y, current.y])
            else:
                line_candidate.set_data([], [])
        else:
            scatter_current.set_data([], [])
            line_candidate.set_data([], [])

        step_desc = f"Step {frame_idx + 1} / {len(steps)}"
        if current is not None:
            step_desc += "\nConsidering point (pop if right turn)"
        else:
            step_desc += "\nPush done / idle"
        text_step.set_text(step_desc)

        return scatter_all, scatter_base, scatter_current, line_hull, line_candidate, text_step

    ax.legend(loc="lower left", fontsize=9)
    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(steps) + 5,
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
    n = 20
    points = [Point2D(float(x), float(y)) for x, y in np.random.rand(n, 2) * 8 + 1]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    gif_path = os.path.join(script_dir, "graham_scan.gif")
    print("Starting Graham Scan animation (close window to exit)")
    create_animation(points, interval=150, save_path=gif_path)
