"""
Chapter 13: 3D Gift Wrapping animation
- Horizon (edge) based: pick an edge, find next face (a,b,p), add face and update horizon.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (projection="3d" registration)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
from typing import List, Tuple, Set, Generator, Optional

# ---------------------------------------------------------------------------
# 3D point and geometry
# ---------------------------------------------------------------------------


class Point3D:
    def __init__(self, x: float, y: float, z: float):
        self.x, self.y, self.z = x, y, z

    def __eq__(self, other: "Point3D") -> bool:
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __hash__(self):
        return hash((self.x, self.y, self.z))

    def __repr__(self):
        return f"({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


def cross3d(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """3D cross product a × b."""
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def dot3d(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def signed_volume(
    a: Tuple[float, float, float],
    b: Tuple[float, float, float],
    c: Tuple[float, float, float],
    d: Tuple[float, float, float],
) -> float:
    """Six times signed volume of tetrahedron ABCD; positive when d is on the positive side of plane (a,b,c)."""
    ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    ad = (d[0] - a[0], d[1] - a[1], d[2] - a[2])
    return dot3d(cross3d(ab, ac), ad)


def _edge(i: int, j: int) -> frozenset:
    return frozenset({i, j})


# ---------------------------------------------------------------------------
# 3D Gift Wrapping: find next face for edge (a,b)
# ---------------------------------------------------------------------------


def _find_next_face(
    points: List[Point3D],
    ai: int,
    bi: int,
) -> Optional[int]:
    """
    Return index of third vertex for next face from edge (points[ai], points[bi]).
    Choose p so all points lie on the same (negative) side of plane (a,b,p). Return None if none.
    """
    a = points[ai].as_tuple()
    b = points[bi].as_tuple()
    n = len(points)
    best: Optional[int] = None
    for p_idx in range(n):
        if p_idx == ai or p_idx == bi:
            continue
        p = points[p_idx].as_tuple()
        # All points on negative side or on plane (a,b,p)?
        valid = True
        for q_idx in range(n):
            if q_idx == ai or q_idx == bi or q_idx == p_idx:
                continue
            q = points[q_idx].as_tuple()
            vol = signed_volume(a, b, p, q)
            if vol > 1e-12:  # point on positive side
                valid = False
                break
        if not valid:
            continue
        if best is None:
            best = p_idx
        else:
            # Tie-break on same plane: pick point farther (outside)
            vol_best = signed_volume(a, b, points[best].as_tuple(), p)
            if vol_best < -1e-12:
                best = p_idx
            elif abs(vol_best) <= 1e-12:
                # Same plane: tie-break by distance
                da = (points[best].x - a[0]) ** 2 + (points[best].y - a[1]) ** 2 + (points[best].z - a[2]) ** 2
                db = (points[p_idx].x - a[0]) ** 2 + (points[p_idx].y - a[1]) ** 2 + (points[p_idx].z - a[2]) ** 2
                if db > da:
                    best = p_idx
    return best


def _make_initial_face(points: List[Point3D]) -> Optional[Tuple[int, int, int]]:
    """Build initial face from three non-collinear points; orient normal outward (all others on negative side)."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = points[i].as_tuple()
                b = points[j].as_tuple()
                c = points[k].as_tuple()
                ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
                ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
                cross_ab_ac = cross3d(ab, ac)
                if dot3d(cross_ab_ac, cross_ab_ac) < 1e-24:
                    continue  # collinear (cross product zero)
                # Orient normal outward: all other points on negative side
                flip = 1
                for q_idx in range(n):
                    if q_idx in (i, j, k):
                        continue
                    q = points[q_idx].as_tuple()
                    v = signed_volume(a, b, c, q)
                    if v > 1e-12:
                        flip = -1
                        break
                    if v < -1e-12:
                        break
                if flip == -1:
                    # (a,c,b) flips normal
                    return (i, k, j)
                return (i, j, k)
    return None


# ---------------------------------------------------------------------------
# 3D Gift Wrapping: yield state for animation
# ---------------------------------------------------------------------------

# State: (faces, current_edge, candidate_idx, horizon_edges, message)
# faces: list of (i,j,k) index triples
# current_edge: (i,j) or None
# candidate_idx: index of point being tested or chosen, or None
# horizon_edges: set of frozenset({i,j})


def gift_wrapping_3d_steps(
    points: List[Point3D],
) -> Generator[
    Tuple[
        List[Tuple[int, int, int]],
        Optional[Tuple[int, int]],
        Optional[int],
        Set[frozenset],
        str,
    ],
    None,
    None,
]:
    """
    Yield (faces, current_edge, candidate_idx, horizon_edges, message) at each step.
    """
    n = len(points)
    if n < 4:
        return

    # Initial face
    face0 = _make_initial_face(points)
    if face0 is None:
        yield ([], None, None, set(), "Collinear: cannot form initial face")
        return

    i0, j0, k0 = face0
    faces: List[Tuple[int, int, int]] = [face0]
    # Edges are undirected: frozenset({i,j}). Horizon = edges with exactly one face
    edge_face_count: dict = {}
    for (a, b, c) in [face0]:
        for e in [_edge(a, b), _edge(b, c), _edge(c, a)]:
            edge_face_count[e] = edge_face_count.get(e, 0) + 1
    horizon: Set[frozenset] = {_edge(i0, j0), _edge(j0, k0), _edge(k0, i0)}

    yield (list(faces), None, None, set(horizon), "Add initial face")

    while horizon:
        edge = next(iter(horizon))
        (ai, bi) = tuple(edge)
        yield (list(faces), (ai, bi), None, set(horizon), "Find next face from edge")

        p_idx = _find_next_face(points, ai, bi)
        if p_idx is None:
            horizon.discard(edge)
            continue

        yield (list(faces), (ai, bi), p_idx, set(horizon), "Choose third vertex for next face")

        new_face = (ai, bi, p_idx)
        faces.append(new_face)

        # Update edge counts for new face
        for (a, b, c) in [new_face]:
            for e in [_edge(a, b), _edge(b, c), _edge(c, a)]:
                edge_face_count[e] = edge_face_count.get(e, 0) + 1

        # Update horizon: remove edges with two faces, keep edges with one
        horizon.discard(edge)
        for e in [_edge(ai, p_idx), _edge(p_idx, bi)]:
            if edge_face_count.get(e, 0) == 1:
                horizon.add(e)
            else:
                horizon.discard(e)

        yield (list(faces), None, None, set(horizon), "Add face, update horizon")

    yield (list(faces), None, None, set(), "Convex hull complete")


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------


def create_animation(
    points: List[Point3D],
    interval: int = 400,
    save_path: Optional[str] = None,
    elev: float = 22,
    azim_start: float = -70,
):
    steps = list(gift_wrapping_3d_steps(points))
    if not steps:
        return None

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    xs = np.array([p.x for p in points])
    ys = np.array([p.y for p in points])
    zs = np.array([p.z for p in points])
    margin = 0.15 * (max(xs.max() - xs.min(), ys.max() - ys.min(), zs.max() - zs.min()) or 1)
    ax.set_xlim(xs.min() - margin, xs.max() + margin)
    ax.set_ylim(ys.min() - margin, ys.max() + margin)
    ax.set_zlim(zs.min() - margin, zs.max() + margin)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("3D Gift Wrapping — add faces from horizon edges", fontsize=12)

    scatter_all = ax.scatter([], [], [], color="lightgray", s=25, alpha=0.8, label="Points")
    scatter_edge = ax.scatter([], [], [], color="lime", s=120, edgecolors="black", linewidths=2, zorder=10, label="Current edge")
    scatter_cand = ax.scatter([], [], [], color="orange", s=150, edgecolors="black", linewidths=2, zorder=11, label="Next vertex")
    poly_collection = Poly3DCollection([], alpha=0.35, facecolor="cornflowerblue", edgecolor="darkblue", linewidths=1.2)
    ax.add_collection3d(poly_collection)
    text_step = ax.text2D(0.02, 0.98, "", transform=ax.transAxes, fontsize=10, verticalalignment="top")

    def get_face_verts(face: Tuple[int, int, int]) -> np.ndarray:
        return np.array([[points[i].x, points[i].y, points[i].z] for i in face])

    def init():
        scatter_all._offsets3d = ([], [], [])
        scatter_edge._offsets3d = ([], [], [])
        scatter_cand._offsets3d = ([], [], [])
        poly_collection.set_verts([])
        text_step.set_text("")
        return [scatter_all, scatter_edge, scatter_cand, text_step]

    def update(frame_idx):
        if frame_idx >= len(steps):
            return [scatter_all, scatter_edge, scatter_cand, text_step]

        faces, current_edge, candidate_idx, horizon_edges, message = steps[frame_idx]

        scatter_all._offsets3d = (xs, ys, zs)

        if current_edge is not None:
            ai, bi = current_edge
            scatter_edge._offsets3d = ([points[ai].x, points[bi].x], [points[ai].y, points[bi].y], [points[ai].z, points[bi].z])
        else:
            scatter_edge._offsets3d = ([], [], [])

        if candidate_idx is not None:
            p = points[candidate_idx]
            scatter_cand._offsets3d = ([p.x], [p.y], [p.z])
        else:
            scatter_cand._offsets3d = ([], [], [])

        if faces:
            verts = [get_face_verts(f) for f in faces]
            poly_collection.set_verts(verts)
        else:
            poly_collection.set_verts([])

        text_step.set_text(f"Step {frame_idx + 1} / {len(steps)}\n{message}")
        ax.view_init(elev=elev, azim=azim_start + frame_idx * 2)
        return [scatter_all, scatter_edge, scatter_cand, text_step]

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(steps) + 4,
        interval=interval,
        blit=False,
        repeat=True,
    )

    plt.tight_layout()
    if save_path:
        anim.save(save_path, writer="pillow", fps=1000 // max(interval, 50))
        plt.close(fig)
    else:
        plt.show()
    return anim


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(13)
    n = 20
    points = [
        Point3D(float(x), float(y), float(z))
        for (x, y, z) in np.random.rand(n, 3) * 4 + 1
    ]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    gif_path = os.path.join(script_dir, "gift_wrapping_3d.gif")
    print("Generating 3D Gift Wrapping animation (close window to exit)")
    create_animation(points, interval=500, save_path=gif_path)
