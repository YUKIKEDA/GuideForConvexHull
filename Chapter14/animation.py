"""
Chapter 14: 3D QuickHull animation
- Initial tetrahedron, then for each face find farthest point, remove visible faces,
  add new faces from horizon, recurse on points outside new faces.
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
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def dot3d(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def signed_volume(
    a: Tuple[float, float, float],
    b: Tuple[float, float, float],
    c: Tuple[float, float, float],
    d: Tuple[float, float, float],
) -> float:
    ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    ad = (d[0] - a[0], d[1] - a[1], d[2] - a[2])
    return dot3d(cross3d(ab, ac), ad)


def _edge(i: int, j: int) -> frozenset:
    return frozenset({i, j})


# ---------------------------------------------------------------------------
# Initial tetrahedron: 4 non-coplanar points, faces with outward normals
# ---------------------------------------------------------------------------


def _make_initial_tetrahedron(points: List[Point3D]) -> Optional[Tuple[List[Tuple[int, int, int]], Set[int]]]:
    """Return (list of 4 faces, set of 4 vertex indices) or None."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a, b, c = points[i].as_tuple(), points[j].as_tuple(), points[k].as_tuple()
                ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
                ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
                if dot3d(cross3d(ab, ac), cross3d(ab, ac)) < 1e-24:
                    continue
                for m in range(n):
                    if m in (i, j, k):
                        continue
                    d = points[m].as_tuple()
                    vol = signed_volume(a, b, c, d)
                    if abs(vol) < 1e-24:
                        continue
                    # Orient so 4th vertex is on negative side of each face (inside)
                    i0, i1, i2, i3 = i, j, k, m
                    if vol > 0:
                        i1, i2 = i2, i1  # swap so vol(0,1,2,3) < 0
                    # Faces: (i0,i1,i2) with i3 inside, (i0,i3,i1) with i2 inside, (i0,i2,i3) with i1 inside, (i1,i3,i2) with i0 inside
                    f1 = (i0, i1, i2)
                    f2 = (i0, i3, i1)
                    f3 = (i0, i2, i3)
                    f4 = (i1, i3, i2)
                    return ([f1, f2, f3, f4], {i0, i1, i2, i3})
    return None


# ---------------------------------------------------------------------------
# Farthest point for a face (max signed_volume among points outside)
# ---------------------------------------------------------------------------


def _farthest_for_face(
    points: List[Point3D],
    face: Tuple[int, int, int],
    outside_indices: Set[int],
) -> Optional[int]:
    ai, bi, ci = face
    a, b, c = points[ai].as_tuple(), points[bi].as_tuple(), points[ci].as_tuple()
    best_idx: Optional[int] = None
    best_vol = -1.0
    for idx in outside_indices:
        if idx in (ai, bi, ci):
            continue
        p = points[idx].as_tuple()
        vol = signed_volume(a, b, c, p)
        if vol > 1e-12 and vol > best_vol:
            best_vol = vol
            best_idx = idx
    return best_idx


def _outside_face(points: List[Point3D], face: Tuple[int, int, int], candidate_indices: Set[int]) -> Set[int]:
    ai, bi, ci = face
    a, b, c = points[ai].as_tuple(), points[bi].as_tuple(), points[ci].as_tuple()
    out: Set[int] = set()
    for idx in candidate_indices:
        if idx in (ai, bi, ci):
            continue
        if signed_volume(a, b, c, points[idx].as_tuple()) > 1e-12:
            out.add(idx)
    return out


# ---------------------------------------------------------------------------
# Visible faces from point p; horizon edges; new faces from horizon + p
# ---------------------------------------------------------------------------


def _visible_faces(points: List[Point3D], faces: List[Tuple[int, int, int]], p_idx: int) -> Set[int]:
    visible: Set[int] = set()
    for fi, f in enumerate(faces):
        ai, bi, ci = f
        if signed_volume(
            points[ai].as_tuple(), points[bi].as_tuple(), points[ci].as_tuple(), points[p_idx].as_tuple()
        ) > 1e-12:
            visible.add(fi)
    return visible


def _horizon_edges(faces: List[Tuple[int, int, int]], visible: Set[int]) -> Set[frozenset]:
    edge_visible_count: dict = {}
    for fi, f in enumerate(faces):
        ai, bi, ci = f
        for e in [_edge(ai, bi), _edge(bi, ci), _edge(ci, ai)]:
            if fi in visible:
                edge_visible_count[e] = edge_visible_count.get(e, 0) + 1
    horizon = {e for e, c in edge_visible_count.items() if c == 1}
    return horizon


def _new_faces_from_horizon(
    points: List[Point3D],
    faces: List[Tuple[int, int, int]],
    visible: Set[int],
    horizon: Set[frozenset],
    p_idx: int,
) -> List[Tuple[int, int, int]]:
    """Build new faces (a,b,p) for each horizon edge; orient so hull stays on negative side."""
    result: List[Tuple[int, int, int]] = []
    a_pt = points[p_idx].as_tuple()
    for e in horizon:
        ai, bi = tuple(e)
        a, b = points[ai].as_tuple(), points[bi].as_tuple()
        for fi, f in enumerate(faces):
            if fi in visible:
                continue
            ai2, bi2, ci2 = f
            if {ai, bi} != {ai2, bi2} and {ai, bi} != {bi2, ci2} and {ai, bi} != {ci2, ai2}:
                continue
            other = next(x for x in (ai2, bi2, ci2) if x not in (ai, bi))
            y = points[other].as_tuple()
            vol = signed_volume(a, b, a_pt, y)
            if vol < -1e-12:
                result.append((ai, bi, p_idx))
            else:
                result.append((ai, p_idx, bi))
            break
        else:
            result.append((ai, bi, p_idx))
    return result


# ---------------------------------------------------------------------------
# 3D QuickHull: yield state for animation (queue-based)
# ---------------------------------------------------------------------------
# State: (faces, current_face, region_indices, farthest_idx, message)


def quick_hull_3d_steps(
    points: List[Point3D],
) -> Generator[
    Tuple[
        List[Tuple[int, int, int]],
        Optional[Tuple[int, int, int]],
        Set[int],
        Optional[int],
        str,
    ],
    None,
    None,
]:
    n = len(points)
    if n < 4:
        return
    tet = _make_initial_tetrahedron(points)
    if tet is None:
        yield ([], None, set(), None, "Cannot form initial tetrahedron")
        return
    faces, _ = tet
    yield (list(faces), None, set(), None, "Initial tetrahedron")
    all_indices = set(range(n))
    queue: List[Tuple[int, Set[int]]] = []
    for fi in range(len(faces)):
        out = _outside_face(points, faces[fi], all_indices)
        if out:
            queue.append((fi, out))
    while queue:
        fi, outside = queue.pop(0)
        if fi >= len(faces) or not outside:
            continue
        face = faces[fi]
        yield (list(faces), face, outside, None, "Process face, find farthest")
        farthest = _farthest_for_face(points, face, outside)
        if farthest is None:
            continue
        yield (list(faces), face, outside, farthest, "Farthest point chosen")
        visible = _visible_faces(points, faces, farthest)
        if not visible:
            continue
        horizon = _horizon_edges(faces, visible)
        new_faces = _new_faces_from_horizon(points, faces, visible, horizon, farthest)
        faces = [f for i, f in enumerate(faces) if i not in visible] + new_faces
        yield (list(faces), None, set(), None, "Visible removed, new faces added")
        base = len(faces) - len(new_faces)
        for i, nf in enumerate(new_faces):
            out_n = _outside_face(points, nf, all_indices)
            if out_n:
                queue.append((base + i, out_n))
    yield (list(faces), None, set(), None, "Convex hull complete")


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
    steps = list(quick_hull_3d_steps(points))
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
    ax.set_title("3D QuickHull — farthest point, visible faces, new faces", fontsize=12)

    scatter_all = ax.scatter([], [], [], color="lightgray", s=25, alpha=0.8, label="Points")
    scatter_region = ax.scatter([], [], [], color="yellow", s=40, alpha=0.9, label="Outside")
    scatter_farthest = ax.scatter([], [], [], color="orange", s=150, edgecolors="black", linewidths=2, zorder=11, label="Farthest")
    poly_collection = Poly3DCollection([], alpha=0.35, facecolor="cornflowerblue", edgecolor="darkblue", linewidths=1.2)
    ax.add_collection3d(poly_collection)
    text_step = ax.text2D(0.02, 0.98, "", transform=ax.transAxes, fontsize=10, verticalalignment="top")

    def get_face_verts(face: Tuple[int, int, int]) -> np.ndarray:
        return np.array([[points[i].x, points[i].y, points[i].z] for i in face])

    def init():
        scatter_all._offsets3d = ([], [], [])
        scatter_region._offsets3d = ([], [], [])
        scatter_farthest._offsets3d = ([], [], [])
        poly_collection.set_verts([])
        text_step.set_text("")
        return [scatter_all, scatter_region, scatter_farthest, text_step]

    def update(frame_idx):
        if frame_idx >= len(steps):
            return [scatter_all, scatter_region, scatter_farthest, text_step]

        faces, current_face, region_indices, farthest_idx, message = steps[frame_idx]

        scatter_all._offsets3d = (xs, ys, zs)

        if region_indices:
            rx = [points[i].x for i in region_indices]
            ry = [points[i].y for i in region_indices]
            rz = [points[i].z for i in region_indices]
            scatter_region._offsets3d = (rx, ry, rz)
        else:
            scatter_region._offsets3d = ([], [], [])

        if farthest_idx is not None:
            p = points[farthest_idx]
            scatter_farthest._offsets3d = ([p.x], [p.y], [p.z])
        else:
            scatter_farthest._offsets3d = ([], [], [])

        if faces:
            verts = [get_face_verts(f) for f in faces]
            poly_collection.set_verts(verts)
        else:
            poly_collection.set_verts([])

        text_step.set_text(f"Step {frame_idx + 1} / {len(steps)}\n{message}")
        ax.view_init(elev=elev, azim=azim_start + frame_idx * 2)
        return [scatter_all, scatter_region, scatter_farthest, text_step]

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
    np.random.seed(14)
    n = 18
    points = [
        Point3D(float(x), float(y), float(z))
        for (x, y, z) in np.random.rand(n, 3) * 4 + 1
    ]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    gif_path = os.path.join(script_dir, "quickhull_3d.gif")
    print("Generating 3D QuickHull animation (close window to exit)")
    create_animation(points, interval=500, save_path=gif_path)
