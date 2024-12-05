import numpy as np
from scipy.linalg import eigvals, solve
import unittest
import numpy as np
from scipy.linalg import det

def vec3cubicroots(a, force_real=False):
    """
    Solve the cubic equation: x^3 + a[2]*x^2 + a[1]*x + a[0] = 0
    Returns the roots and the number of real solutions.
    """
    r = np.zeros(3)
    c1 = a[1] - a[2] ** 2 / 3
    c0 = a[0] - a[1] * a[2] / 3 + 2 / 27 * a[2] ** 3

    if c1 == 0:
        if c0 == 0:
            r[0] = 0
        elif c0 > 0:
            r[0] = -np.cbrt(c0)
        else:
            r[0] = np.cbrt(-c0)
    else:
        neg_c1 = c1 < 0
        abs_c1 = abs(c1)
        k = np.sqrt(4 / 3 * abs_c1)
        d0 = c0 * 4 / (k ** 3)

        if neg_c1:
            if d0 > 1:
                r[0] = -np.cosh(np.arccosh(d0) / 3)
            elif d0 > -1:
                r[0] = -np.cos(np.arccos(d0) / 3)
            else:
                r[0] = np.cosh(np.arccosh(-d0) / 3)
        else:
            r[0] = -np.sinh(np.arcsinh(d0) / 3)

        r[0] *= k
    r[0] -= a[2] / 3

    # Other two solutions
    p = r[0] + a[2]
    q = r[0] * p + a[1]
    discrim = p ** 2 - 4 * q
    if force_real and discrim < 0:
        discrim = 0

    if discrim >= 0:
        root = np.sqrt(discrim)
        r[1] = (-p - root) / 2
        r[2] = (-p + root) / 2
        return 3, r
    else:
        root = np.sqrt(-discrim)
        r[1] = -p / 2
        r[2] = root / 2
        return 1, r


def mat3invariants(m):
    """
    Compute the invariants of a 3x3 matrix.
    """
    invariant0 = -np.linalg.det(m)
    invariant1 = (m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1]
                  + m[2, 2] * m[0, 0] - m[2, 0] * m[0, 2]
                  + m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0])
    invariant2 = -(m[0, 0] + m[1, 1] + m[2, 2])
    return np.array([invariant0, invariant1, invariant2])


def mat3eigenvalues(m):
    """
    Compute eigenvalues of a 3x3 matrix and return the number of real eigenvalues.
    """
    pqr = mat3invariants(m)
    num_real, eigenvalues = vec3cubicroots(pqr, force_real=True)
    return num_real, eigenvalues


def mat3real_eigenvector(m, lambda_val):
    """
    Compute the eigenvector corresponding to a real eigenvalue of a 3x3 matrix.
    """
    reduced = m - lambda_val * np.eye(3)
    cross = np.array([np.cross(reduced[1], reduced[2]),
                      np.cross(reduced[2], reduced[0]),
                      np.cross(reduced[0], reduced[1])])
    norms = np.array([np.linalg.norm(cross[i]) for i in range(3)])

    # Use the largest cross product to calculate the eigenvector
    best = np.argmax(norms)
    if norms[best] > 0:
        return cross[best] / norms[best], True
    return None, False


def solve_parallel_vectors(v, w, dimensions, origin, spacing, feature, feature_strength):
    """
    Solve parallel vectors field with implicit, bifurcation, or vortex core feature.
    """
    dim_x, dim_y, dim_z = dimensions
    spacing_x, spacing_y, spacing_z = spacing
    origin_x, origin_y, origin_z = origin

    num_nodes = dim_x * dim_y * dim_z
    result = np.full((num_nodes, 3), np.nan)

    for idx in range(num_nodes):
        i = idx % dim_x
        j = (idx // dim_x) % dim_y
        k = (idx // (dim_x * dim_y))

        if i >= dim_x - 1 or j >= dim_y - 1 or k >= dim_z - 1:
            continue

        # Extract local vectors for computation
        v_cell = np.array([
            v[i, j, k],
            v[i + 1, j, k],
            v[i, j + 1, k],
            v[i + 1, j + 1, k]
        ])

        w_cell = np.array([
            w[i, j, k],
            w[i + 1, j, k],
            w[i, j + 1, k],
            w[i + 1, j + 1, k]
        ])

        # Compute Jacobian and other matrix-related quantities
        matrix = np.random.random((3, 3))  # Example placeholder, replace with actual calculation

        # Eigenvalue computation for features
        num_real, eigenvalues = mat3eigenvalues(matrix)
        if num_real == 1:
            result[idx] = eigenvalues[0]  # Replace with more detailed logic as needed

    return result



class TestMathFunctions(unittest.TestCase):

    def test_vec3cubicroots(self):
        a = [6, 11, 6]  # Coefficients for x^3 + 6x^2 + 11x + 6 = 0
        expected_real_roots = [-1, -2, -3]
        num_real, roots = vec3cubicroots(a, force_real=True)
        self.assertEqual(num_real, 3)
        np.testing.assert_almost_equal(sorted(roots), sorted(expected_real_roots), decimal=6)

    def test_mat3invariants(self):
        m = np.array([
            [1, 2, 3],
            [0, 4, 5],
            [1, 0, 6]
        ])
        invariant0 = -det(m)  # Determinant
        invariant1 = (m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1] +
                      m[2, 2] * m[0, 0] - m[2, 0] * m[0, 2] +
                      m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0])
        invariant2 = -(m[0, 0] + m[1, 1] + m[2, 2])  # Trace
        expected = np.array([invariant0, invariant1, invariant2])

        result = mat3invariants(m)
        np.testing.assert_almost_equal(result, expected, decimal=6)

    def test_mat3eigenvalues(self):
        m = np.array([
            [4, 1, 2],
            [1, 3, 0],
            [2, 0, 5]
        ])
        expected_eigenvalues = np.sort(eigvals(m).real)
        num_real, eigenvalues = mat3eigenvalues(m)
        self.assertEqual(num_real, 3)
        np.testing.assert_almost_equal(np.sort(eigenvalues), expected_eigenvalues, decimal=6)

    def test_mat3real_eigenvector(self):
        m = np.array([
            [4, 1, 2],
            [1, 3, 0],
            [2, 0, 5]
        ])
        eigenvalues = eigvals(m).real
        for lambda_val in eigenvalues:
            ev, valid = mat3real_eigenvector(m, lambda_val)
            self.assertTrue(valid)
            np.testing.assert_almost_equal(np.dot(m, ev), lambda_val * ev, decimal=6)

    def test_solve_parallel_vectors(self):
        dimensions = (3, 3, 3)
        origin = (0, 0, 0)
        spacing = (1, 1, 1)
        v = np.random.random((3, 3, 3, 3))
        w = np.random.random((3, 3, 3, 3))

        result = solve_parallel_vectors(v, w, dimensions, origin, spacing, feature=1, feature_strength=0.1)
        self.assertEqual(result.shape, (27, 3))  # Dimensions match expected output
        # Additional validations could be added here based on known input/output

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import sys
    sys.path.insert(0, "/home/max/Repos/Magnetic-Reconnection-Visualization/src")
    from mrvis.algorithms import *
    from mrvis.utils import save_as_vti

    size = 64
    dimensions = (size, size, size)

    xmin, xmax = -3, 3
    ymin, ymax = -1, 4
    zmin, zmax = -3, 3

    linx = np.linspace(xmin, xmax, size)
    liny = np.linspace(ymin, ymax, size)
    linz = np.linspace(zmin, zmax, size)

    origin = [linx[0], liny[0], linz[0]]
    spacing = [linx[1] - linx[0], liny[1] - liny[0], linz[1] - linz[0]]

    x, y, z = np.meshgrid(linx, liny, linz, indexing="ij")

    field = np.array([
        (y -2) ** 2 - 1 + z**2,
        -x,
        np.ones(x.shape) * 1.0
    ])

    result = convective_acceleration(field, linx, liny, linz)
    
    field = field.transpose((1, 2, 3, 0))
    result = result.transpose((1, 2, 3, 0))
    
    result = solve_parallel_vectors(field, result, dimensions, origin, spacing, feature=1, feature_strength=0.1) 
    print(result)
    unittest.main()



    
