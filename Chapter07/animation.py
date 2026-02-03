"""
Chapter 7: QuickHull (Divide and Conquer) animation
- Split by line L-R, find farthest point P, recurse on left-of-A-P and left-of-P-B.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon
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


def quick_hull(points: List[Point2D]) -> List[Point2D]:
    """Return convex hull (CCW). Used for final frame."""
    n = len(points)
    if n < 3:
        return list(points)
    sorted_pts = sorted(points, key=lambda p: (p.x, p.y))
    left, right = sorted_pts[0], sorted_pts[-1]
    upper = [p for p in sorted_pts[1:-1] if cross(left, right, p) > 0]
    lower = [p for p in sorted_pts[1:-1] if cross(left, right, p) < 0]

    def _find_hull(S: List[Point2D], a: Point2D, b: Point2D) -> List[Point2D]:
        if not S:
            return []
        P = max(S, key=lambda p: cross(a, b, p))
        s1 = [p for p in S if p != P and cross(a, P, p) > 0]
        s2 = [p for p in S if p != P and cross(P, b, p) > 0]
        hull1 = _find_hull(s1, a, P)
        hull2 = _find_hull(s2, P, b)
        return hull1 + [P] + hull2

    hull_upper = _find_hull(upper, left, right)
    hull_lower = _find_hull(lower, right, left)
    return [left] + hull_upper + [right] + hull_lower


# ---------------------------------------------------------------------------
# QuickHull: yield state at each recursion step
# ---------------------------------------------------------------------------

# State: (segment_a, segment_b, region_points, farthest, full_hull_or_None)
# full_hull_or_None is set only on the final frame.


def _find_hull_steps(
    points: List[Point2D],
    a: Point2D,
    b: Point2D,
) -> Generator[Tuple[Point2D | None, Point2D | None, List[Point2D], Point2D | None, List[Point2D] | None], None, None]:
    """Yield (a, b, region_points, farthest, None) for each step. No final hull here."""
    yield (a, b, list(points), None, None)
    if not points:
        return
    farthest = max(points, key=lambda p: cross(a, b, p))
    yield (a, b, list(points), farthest, None)
    s1 = [p for p in points if p != farthest and cross(a, farthest, p) > 0]
    s2 = [p for p in points if p != farthest and cross(farthest, b, p) > 0]
    yield from _find_hull_steps(s1, a, farthest)
    yield from _find_hull_steps(s2, farthest, b)


def quick_hull_steps(
    points: List[Point2D],
) -> Generator[Tuple[Point2D | None, Point2D | None, List[Point2D], Point2D | None, List[Point2D] | None], None, None]:
    """
    Yield (a, b, region_points, farthest, full_hull_or_None) at each step.
    - a, b: current segment (edge we're processing). None on final frame.
    - region_points: points in current region (left of segment a-b).
    - farthest: farthest point from segment (None when just entered or on final frame).
    - full_hull_or_None: convex hull of all points, only on final frame.
    """
    n = len(points)
    if n == 0:
        return
    if n < 3:
        yield (None, None, list(points), None, list(points))
        return

    sorted_pts = sorted(points, key=lambda p: (p.x, p.y))
    left, right = sorted_pts[0], sorted_pts[-1]
    upper = [p for p in sorted_pts[1:-1] if cross(left, right, p) > 0]
    lower = [p for p in sorted_pts[1:-1] if cross(left, right, p) < 0]

    # Top-level: upper half
    yield (left, right, list(upper), None, None)
    if upper:
        farthest_upper = max(upper, key=lambda p: cross(left, right, p))
        yield (left, right, list(upper), farthest_upper, None)
    yield from _find_hull_steps(upper, left, right)

    # Top-level: lower half
    yield (right, left, list(lower), None, None)
    if lower:
        farthest_lower = max(lower, key=lambda p: cross(right, left, p))
        yield (right, left, list(lower), farthest_lower, None)
    yield from _find_hull_steps(lower, right, left)

    # Final frame: full hull
    full_hull = quick_hull(points)
    yield (None, None, [], None, full_hull)


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------


def create_animation(
    points: List[Point2D],
    interval: int = 120,
    save_path: str | None = None,
):
    steps = list(quick_hull_steps(points))
    if not steps:
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_aspect("equal")
    ax.set_title("QuickHull (Divide and Conquer) — 2D Convex Hull", fontsize=14)

    xs = [p.x for p in points]
    ys = [p.y for p in points]
    margin = 0.1 * (max(max(xs) - min(xs), max(ys) - min(ys)) or 1)
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)

    (scatter_all,) = ax.plot([], [], "o", color="lightgray", ms=8, label="Points")
    (scatter_region,) = ax.plot([], [], "o", color="steelblue", ms=8, label="Region")
    (scatter_farthest,) = ax.plot(
        [], [], "o", color="orange", ms=14, markeredgewidth=2, markeredgecolor="black", zorder=5, label="Farthest"
    )
    (line_segment,) = ax.plot([], [], "g-", lw=2, label="Segment")
    (line_hull,) = ax.plot([], [], "b-", lw=2.5, label="Hull")
    triangle_patch = ax.add_patch(Polygon([[0, 0], [0, 0], [0, 0]], facecolor="green", alpha=0.15, edgecolor="none", label="Triangle L-P-R"))
    triangle_patch.set_visible(False)
    text_step = ax.text(0.02, 0.98, "", transform=ax.transAxes, fontsize=11, verticalalignment="top")

    def init():
        scatter_all.set_data([], [])
        scatter_region.set_data([], [])
        scatter_farthest.set_data([], [])
        line_segment.set_data([], [])
        line_hull.set_data([], [])
        triangle_patch.set_visible(False)
        text_step.set_text("")
        return scatter_all, scatter_region, scatter_farthest, line_segment, line_hull, triangle_patch, text_step

    def update(frame_idx):
        if frame_idx >= len(steps):
            return scatter_all, scatter_region, scatter_farthest, line_segment, line_hull, triangle_patch, text_step

        a, b, region_points, farthest, full_hull = steps[frame_idx]

        scatter_all.set_data(xs, ys)

        if region_points:
            rx = [q.x for q in region_points]
            ry = [q.y for q in region_points]
            scatter_region.set_data(rx, ry)
        else:
            scatter_region.set_data([], [])

        if farthest is not None:
            scatter_farthest.set_data([farthest.x], [farthest.y])
        else:
            scatter_farthest.set_data([], [])

        if a is not None and b is not None:
            line_segment.set_data([a.x, b.x], [a.y, b.y])
        else:
            line_segment.set_data([], [])

        if farthest is not None and a is not None and b is not None:
            tri_xy = np.array([[a.x, a.y], [farthest.x, farthest.y], [b.x, b.y]])
            triangle_patch.set_xy(tri_xy)
            triangle_patch.set_visible(True)
        else:
            triangle_patch.set_visible(False)

        if full_hull is not None and len(full_hull) >= 2:
            hx = [q.x for q in full_hull] + [full_hull[0].x]
            hy = [q.y for q in full_hull] + [full_hull[0].y]
            line_hull.set_data(hx, hy)
        else:
            line_hull.set_data([], [])

        step_desc = f"Step {frame_idx + 1} / {len(steps)}"
        if full_hull is not None:
            step_desc += "\nHull complete"
        elif farthest is not None:
            step_desc += "\nFarthest point (triangle: points inside discarded)"
        else:
            step_desc += "\nSegment and region"
        text_step.set_text(step_desc)

        return scatter_all, scatter_region, scatter_farthest, line_segment, line_hull, triangle_patch, text_step

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
    n = 16
    points = [Point2D(float(x), float(y)) for x, y in np.random.rand(n, 2) * 8 + 1]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    gif_path = os.path.join(script_dir, "quickhull.gif")
    print("Starting QuickHull animation (close window to exit)")
    create_animation(points, interval=180, save_path=gif_path)
