"""
Chapter 5: Andrew's Monotone Chain animation
- Sort by (x, y); build lower hull (left to right), then upper hull (right to left); combine.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Tuple, Generator

# ---------------------------------------------------------------------------
# 2D point and geometry (same as Chapter 2/3/4)
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


# ---------------------------------------------------------------------------
# Andrew's Monotone Chain: yield state for animation
# ---------------------------------------------------------------------------


def andrew_monotone_chain_steps(
    points: List[Point2D],
) -> Generator[Tuple[str, List[Point2D], List[Point2D], Point2D | None], None, None]:
    """
    Yield (phase, lower_hull, upper_hull, current_point) at each step.
    - phase: "lower" | "upper" | "done"
    - lower_hull: current lower chain (left to right).
    - upper_hull: current upper chain (built right to left); only during "upper".
    - current_point: point we are considering to add (None when idle or after push).
    """
    n = len(points)
    if n == 0:
        return
    if n < 3:
        yield ("done", list(points), [], None)
        return

    sorted_pts = sorted(points, key=lambda p: (p.x, p.y))

    # Lower hull (left to right)
    lower: List[Point2D] = []
    yield ("lower", list(lower), [], None)
    for p in sorted_pts:
        yield ("lower", list(lower), [], p)
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
            yield ("lower", list(lower), [], p)
        lower.append(p)
        yield ("lower", list(lower), [], None)

    # Upper hull (right to left)
    upper: List[Point2D] = []
    yield ("upper", list(lower), list(upper), None)
    for p in reversed(sorted_pts):
        yield ("upper", list(lower), list(upper), p)
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
            yield ("upper", list(lower), list(upper), p)
        upper.append(p)
        yield ("upper", list(lower), list(upper), None)

    # Done: combined hull (lower[:-1] + upper[:-1])
    yield ("done", list(lower), list(upper), None)


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------


def create_animation(
    points: List[Point2D],
    interval: int = 120,
    save_path: str | None = None,
):
    steps = list(andrew_monotone_chain_steps(points))
    if not steps:
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_aspect("equal")
    ax.set_title("Andrew's Monotone Chain — 2D Convex Hull", fontsize=14)

    xs = [p.x for p in points]
    ys = [p.y for p in points]
    margin = 0.1 * (max(max(xs) - min(xs), max(ys) - min(ys)) or 1)
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)

    (scatter_all,) = ax.plot([], [], "o", color="lightgray", ms=8, label="Points")
    (scatter_current,) = ax.plot(
        [], [], "o", color="orange", ms=12, markeredgewidth=2, markeredgecolor="black", zorder=5, label="Current"
    )
    (line_lower,) = ax.plot([], [], "b-", lw=2.5, label="Lower hull")
    (line_upper,) = ax.plot([], [], "green", lw=2.5, label="Upper hull")
    (line_candidate,) = ax.plot([], [], "m--", lw=1.5, alpha=0.8, label="To current")
    text_step = ax.text(0.02, 0.98, "", transform=ax.transAxes, fontsize=11, verticalalignment="top")

    def init():
        scatter_all.set_data([], [])
        scatter_current.set_data([], [])
        line_lower.set_data([], [])
        line_upper.set_data([], [])
        line_candidate.set_data([], [])
        text_step.set_text("")
        return scatter_all, scatter_current, line_lower, line_upper, line_candidate, text_step

    def update(frame_idx):
        if frame_idx >= len(steps):
            return scatter_all, scatter_current, line_lower, line_upper, line_candidate, text_step

        phase, lower, upper, current = steps[frame_idx]

        scatter_all.set_data(xs, ys)

        if lower:
            lx = [q.x for q in lower]
            ly = [q.y for q in lower]
            line_lower.set_data(lx, ly)
        else:
            line_lower.set_data([], [])

        if upper:
            ux = [q.x for q in upper]
            uy = [q.y for q in upper]
            line_upper.set_data(ux, uy)
        else:
            line_upper.set_data([], [])

        if current is not None:
            scatter_current.set_data([current.x], [current.y])
            # Line from top of current chain to current point
            if phase == "lower" and lower:
                line_candidate.set_data([lower[-1].x, current.x], [lower[-1].y, current.y])
            elif phase == "upper" and upper:
                line_candidate.set_data([upper[-1].x, current.x], [upper[-1].y, current.y])
            else:
                line_candidate.set_data([], [])
        else:
            scatter_current.set_data([], [])
            line_candidate.set_data([], [])

        step_desc = f"Step {frame_idx + 1} / {len(steps)}"
        if phase == "lower":
            step_desc += "\nBuilding lower hull (left to right)"
        elif phase == "upper":
            step_desc += "\nBuilding upper hull (right to left)"
        else:
            step_desc += "\nHull complete"
        if current is not None:
            step_desc += " — considering point"
        text_step.set_text(step_desc)

        return scatter_all, scatter_current, line_lower, line_upper, line_candidate, text_step

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
    gif_path = os.path.join(script_dir, "andrew_monotone_chain.gif")
    print("Starting Andrew's Monotone Chain animation (close window to exit)")
    create_animation(points, interval=150, save_path=gif_path)
