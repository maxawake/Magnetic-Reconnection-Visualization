import tqdm
import numpy as np
from findiff import Gradient

def curl(dx, dy, dz, u, v, w):
    dFx_dx, dFx_dy, dFx_dz = np.gradient(u, dx, dy, dz)
    dFy_dx, dFy_dy, dFy_dz = np.gradient(v, dx, dy, dz)
    dFz_dx, dFz_dy, dFz_dz = np.gradient(w, dx, dy, dz)

    rot_x = dFz_dy - dFy_dz
    rot_y = dFx_dz - dFz_dx
    rot_z = dFy_dx - dFx_dy

    return np.array([rot_x, rot_y, rot_z])


def jacobian(vec, dx, dy, dz, order=2):
    if order==1:
        dudx, dudy, dudz = np.gradient(vec[0], dx, dy, dz)
        dvdx, dvdy, dvdz = np.gradient(vec[1], dx, dy, dz)
        dwdx, dwdy, dwdz = np.gradient(vec[2], dx, dy, dz)
    elif order==2:
        ddx = dx[1] - dx[0]
        ddy = dy[1] - dy[0]
        ddz = dz[1] - dz[0]
        grad = Gradient(h=[ddx, ddy, ddz], acc=6)
        dudx, dudy, dudz = grad(vec[0])
        dvdx, dvdy, dvdz = grad(vec[1])
        dwdx, dwdy, dwdz = grad(vec[2])

    return np.array([[dudx, dudy, dudz], [dvdx, dvdy, dvdz], [dwdx, dwdy, dwdz]])


def vortex_core(vec, x, y, z):
    """
    Calculate the vortex core of a given vector field with the Sujudi-Haimes criterion
    """
    sizex, sizey, sizez = vec.shape[1], vec.shape[1], vec.shape[2]

    Jac = jacobian(vec, x, y, z)

    vortex_core = np.zeros((sizex, sizey, sizez))
    for i in range(sizex):
        for j in range(sizey):
            for k in range(sizez):
                eigenvals, eigenvecs = np.linalg.eig(Jac[:, :, i, j, k])
                for eigvec in eigenvecs:
                    if (np.linalg.norm(eigvec - vec[:, i, j, k])) < 1.0:
                        vortex_core[i, j, k] = 1

    if np.any(vortex_core) == 1:
        print("Found points")
    else:
        print("No points found!")
    return vortex_core


def convective_acceleration(vec, x, y, z):
    Jac = jacobian(vec, x, y, z)

    # Reshape vec for vectorized operations
    vec_reshaped = vec.reshape(vec.shape[0], -1).T
    Jac_reshaped = Jac.reshape(Jac.shape[0], Jac.shape[1], -1).transpose(2, 0, 1)

    # Calculate directional derivative (grad u)u
    Jv = (Jac_reshaped @ vec_reshaped[..., np.newaxis]).squeeze(-1)

    return Jv.reshape(vec.shape[1], vec.shape[2], vec.shape[3], 3).transpose(3, 0, 1, 2)


def parallel_vector_operator(vec, x, y, z):
    sizex, sizey, sizez = vec.shape[1], vec.shape[2], vec.shape[3]

    Jvv = convective_acceleration(vec, x, y, z).reshape(3, -1)
    vec = vec.reshape(3, -1)

    cross_product = np.cross(vec.T, Jvv.T)
    parallel = np.linalg.norm(cross_product.T, axis=0).reshape(sizex, sizey, sizez)

    return parallel


def shear_layer(flow, dx, dy, dz):
    Jac = jacobian(flow, dx, dy, dz)
    Jac = Jac.reshape(Jac.shape[0], Jac.shape[1], -1).transpose(2, 0, 1)
    S = (Jac + Jac.transpose(0, 2, 1)) / 2
    eigenvals = np.linalg.eigvals(S)
    l1 = (eigenvals[:, 0] - eigenvals[:, 1]) ** 2
    l2 = (eigenvals[:, 0] - eigenvals[:, 2]) ** 2
    l3 = (eigenvals[:, 1] - eigenvals[:, 2]) ** 2
    result = np.sqrt((l1 + l2 + l3) / 6)
    return result.reshape(flow.shape[1], flow.shape[2], flow.shape[3])


def lambda2(flow, dx, dy, dz):
    """Calculate the Lambda2 criterion for a given flow field to identify vortex cores

    Parameters
    ----------
    flow : ndarray
        3D vector field of shape (3, sizex, sizey, sizez)
    dx : ndarray
        Spacing in x-direction of shape (sizex,)
    dy : ndarray
        Spacing in y-direction of shape (sizey,)
    dz : ndarray
        Spacing in z-direction of shape (sizez,)

    Returns
    -------
    ndarray
        Lambda2 criterion of shape (sizex, sizey, sizez)
    """
    n, sizex, sizey, sizez = flow.shape
    Jac = jacobian(flow, dx, dy, dz)
    result = np.zeros((sizex, sizey, sizez))
    Jac_current = Jac.reshape(3, 3, -1).transpose(2, 0, 1)

    # Get symmetric and antisymmetric part of Jacobian (shear and rotation)
    S = (Jac_current + Jac_current.transpose(0, 2, 1)) / 2
    Omega = (Jac_current - Jac_current.transpose(0, 2, 1)) / 2

    # Calculate Lambda2
    M = S @ S + Omega @ Omega
    eigenvals = np.linalg.eigvals(M)
    sorted_vals = np.sort(eigenvals, axis=1)
    result = sorted_vals[:, 1]
    return result.reshape(sizex, sizey, sizez)
