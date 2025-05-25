import numpy as np


# ------------------------------------------------------------------
# The two helper functions copied from the previous answer
# ------------------------------------------------------------------
def parallel_vectors_triangle(positions, v, w, tol=1e-15):
    V = np.empty((3, 3))
    W = np.empty((3, 3))
    V[:, 0] = v[:, 0]
    V[:, 1] = v[:, 1] - v[:, 0]
    V[:, 2] = v[:, 2] - v[:, 0]
    W[:, 0] = w[:, 0]
    W[:, 1] = w[:, 1] - w[:, 0]
    W[:, 2] = w[:, 2] - w[:, 0]

    detV = abs(np.linalg.det(V))
    detW = abs(np.linalg.det(W))
    if detV < tol and detW < tol:
        return np.empty((3, 0))

    X = np.linalg.solve(V, W) if detV >= detW else np.linalg.solve(W, V)

    eigvals = np.linalg.eigvals(X)

    sols = []
    for lam in eigvals:
        if abs(lam.imag) >= tol:
            continue
        lam = lam.real
        reduced = X - lam * np.eye(3)
        cross = np.column_stack(
            [
                np.cross(reduced[1], reduced[2]),
                np.cross(reduced[2], reduced[0]),
                np.cross(reduced[0], reduced[1]),
            ]
        )
        norms = np.sum(cross**2, axis=0)
        best = np.argmax(norms)
        len_best = np.sqrt(norms[best])
        if len_best < tol:
            continue
        ev = cross[:, best] / len_best

        if abs(ev[0]) < tol:
            continue
        s = ev[1] / ev[0]
        t = ev[2] / ev[0]
        if (s < 0.0) or (t < 0.0) or (s + t > 1.0):
            continue

        v_interp = (1.0 - s - t) * v[:, 0] + s * v[:, 1] + t * v[:, 2]
        w_interp = (1.0 - s - t) * w[:, 0] + s * w[:, 1] + t * w[:, 2]
        if np.sum(np.cross(v_interp, w_interp) ** 2) >= tol:
            continue

        p = (1.0 - s - t) * positions[:, 0] + s * positions[:, 1] + t * positions[:, 2]
        sols.append(p)

    return np.column_stack(sols) if sols else np.empty((3, 0))


def parallel_vectors_cell(positions, v, w):
    triangles = np.array(
        [
            [0, 1, 2],
            [1, 3, 2],
            [0, 2, 6],
            [0, 6, 4],
            [0, 1, 4],
            [1, 5, 4],
            [4, 5, 6],
            [5, 7, 6],
            [1, 3, 7],
            [1, 7, 5],
            [2, 3, 7],
            [2, 7, 6],
        ],
        dtype=int,
    )

    sol_pts = []
    sol_faces = []

    for tri_id, tri in enumerate(triangles):
        pts = parallel_vectors_triangle(positions[:, tri], v[:, tri], w[:, tri])
        if pts.size > 0:
            sol_pts.append(pts)
            sol_faces.extend([tri_id // 2] * pts.shape[1])

    if not sol_pts:
        return np.empty((3, 0)), np.empty((0,), dtype=int)

    sol_pts = np.hstack(sol_pts)
    sol_faces = np.asarray(sol_faces, dtype=int)

    n = sol_pts.shape[1]
    taken = np.zeros(n, dtype=bool)
    out_pts = []
    out_faces = []

    for i in range(n):
        if taken[i]:
            continue
        candidates = np.where(~taken & (sol_faces != sol_faces[i]))[0]
        if candidates.size == 0:
            continue
        dists = np.sum((sol_pts[:, i, None] - sol_pts[:, candidates]) ** 2, axis=0)
        j = candidates[np.argmin(dists)]
        taken[i] = taken[j] = True

        out_pts.append(sol_pts[:, i])
        out_pts.append(sol_pts[:, j])
        out_faces.append(sol_faces[i])
        out_faces.append(sol_faces[j])

    if not out_pts:
        return np.empty((3, 0)), np.empty((0,), dtype=int)

    return np.column_stack(out_pts), np.asarray(out_faces, dtype=int)


# ------------------------------------------------------------------
# Volume-wrapper: iterate all cells in a structured grid
# ------------------------------------------------------------------
def parallel_vectors_volume(X, Y, Z, v_field, w_field):
    """
    Parameters
    ----------
    X,Y,Z      : 3-D arrays (nx,ny,nz)  – coordinates
    v_field    : 4-D (3,nx,ny,nz)       – first vector field
    w_field    : 4-D (3,nx,ny,nz)       – second vector field
    Returns
    -------
    all_pts  : (3,M) concatenated PV end-points
    all_src  : (M,) cell indices (flat) to which each point belongs
    """
    nx, ny, nz = X.shape
    all_pts = []
    all_src = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                # gather eight corners in VTK hexahedral order
                corners = [
                    (i, j, k),
                    (i + 1, j, k),
                    (i, j + 1, k),
                    (i + 1, j + 1, k),
                    (i, j, k + 1),
                    (i + 1, j, k + 1),
                    (i, j + 1, k + 1),
                    (i + 1, j + 1, k + 1),
                ]
                pos = np.column_stack([np.array([X[c], Y[c], Z[c]]) for c in corners])
                v = np.column_stack([v_field[:, c[0], c[1], c[2]] for c in corners])
                w = np.column_stack([w_field[:, c[0], c[1], c[2]] for c in corners])

                pts, _ = parallel_vectors_cell(pos, v, w)
                if pts.size:
                    all_pts.append(pts)
                    all_src.extend([(i * (ny - 1) + j) * (nz - 1) + k] * pts.shape[1])

    if not all_pts:
        return np.empty((3, 0)), np.empty((0,), dtype=int)
    return np.hstack(all_pts), np.asarray(all_src, dtype=int)


# ------------------------------------------------------------------
# Example usage with the user's grid
# ------------------------------------------------------------------
xlin = np.linspace(-1, 1, 10)
X, Y, Z = np.meshgrid(xlin, xlin, xlin, indexing="ij")

v_field = np.empty((3,) + X.shape)
v_field[0] = -Y
v_field[1] = X
v_field[2] = Z

# choose a second field with visible PV intersections, e.g. w = (X, Y, -Z)
w_field = np.empty_like(v_field)
w_field[0] = X
w_field[1] = Y
w_field[2] = -Z

pv_pts, pv_cell_ids = parallel_vectors_volume(X, Y, Z, v_field, w_field)

import matplotlib.pyplot as plt

print(f"Total PV end-points: {pv_pts.shape[1]}")
print("First few points:\n", pv_pts[:, : min(6, pv_pts.shape[1])])

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection="3d")

# plot the line segments (pv_pts[:,0], pv_pts[:,1]), (pv_pts[:,2], pv_pts[:,3]), …
for i in range(0, pv_pts.shape[1], 2):
    xs, ys, zs = pv_pts[:, i : i + 2]
    ax.plot(xs, ys, zs)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Parallel-Vector Line Segments")
plt.show()
