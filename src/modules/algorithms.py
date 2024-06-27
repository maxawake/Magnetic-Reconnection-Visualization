import tqdm
import numpy as np


def curl(dx, dy, dz, u, v, w):
    dFx_dx, dFx_dy, dFx_dz = np.gradient(u, dx, dy, dz)
    dFy_dx, dFy_dy, dFy_dz = np.gradient(v, dx, dy, dz)
    dFz_dx, dFz_dy, dFz_dz = np.gradient(w, dx, dy, dz)

    rot_x = dFz_dy - dFy_dz
    rot_y = dFx_dz - dFz_dx
    rot_z = dFy_dx - dFx_dy

    return np.array([rot_x, rot_y, rot_z])


def get_jacobian(vec, dx, dy, dz):
    dudx, dudy, dudz = np.gradient(vec[0], dx, dy, dz)
    dvdx, dvdy, dvdz = np.gradient(vec[1], dx, dy, dz)
    dwdx, dwdy, dwdz = np.gradient(vec[2], dx, dy, dz)

    return np.array([[dudx, dudy, dudz], [dvdx, dvdy, dvdz], [dwdx, dwdy, dwdz]])  # .transpose(1, 0, 2, 3, 4)


def get_vortex_core(vec, x, y, z):
    sizex, sizey, sizez = vec.shape[1], vec.shape[1], vec.shape[2]

    Jac = get_jacobian(vec, x, y, z)

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


def get_directional_derivative(vec, x, y, z):
    Jac = get_jacobian(vec, x, y, z)

    # Reshape vec for vectorized operations
    vec_reshaped = vec.reshape(vec.shape[0], -1).T
    Jac_reshaped = Jac.reshape(Jac.shape[0], Jac.shape[1], -1).transpose(2, 0, 1)

    # Calculate directional derivative (grad u)u
    Jv = (Jac_reshaped @ vec_reshaped[..., np.newaxis]).squeeze(-1)

    return Jv.reshape(vec.shape[1], vec.shape[2], vec.shape[3], 3).transpose(3, 0, 1, 2)


def get_parallel_vector_operator(vec, x, y, z):
    sizex, sizey, sizez = vec.shape[1], vec.shape[2], vec.shape[3]

    Jv = get_directional_derivative(vec, x, y, z)

    cross_product = np.cross(vec.reshape(vec.shape[0], -1).T, Jv.T)
    parallel = np.linalg.norm(cross_product, axis=1).reshape(sizex, sizey, sizez)

    return parallel


def get_lambda2(flow, dx, dy, dz):
    size = dx.shape[0]
    Jac = get_jacobian(flow, dx, dy, dz)
    result = np.zeros((size, size, size))
    for i in tqdm.tqdm(range(size)):
        for j in range(size):
            for k in range(size):
                Jac_current = Jac[:, :, i, j, k]
                S = (Jac_current + Jac_current.T) / 2
                Omega = (Jac_current - Jac_current.T) / 2
                M = S @ S + Omega @ Omega
                eigenvals, eigenvecs = np.linalg.eig(M)
                sorted_vals = np.sort(eigenvals)
                result[i, j, k] = sorted_vals[1]
    return result


def get_shear_layer(flow, Jac, dx, dy, dz):
    Jac = Jac.reshape(Jac.shape[0], Jac.shape[1], -1).transpose(2, 0, 1)
    S = (Jac + Jac.transpose(0, 2, 1)) / 2
    eigenvals = np.linalg.eigvals(S)
    l1 = (eigenvals[:, 0] - eigenvals[:, 1]) ** 2
    l2 = (eigenvals[:, 0] - eigenvals[:, 2]) ** 2
    l3 = (eigenvals[:, 1] - eigenvals[:, 2]) ** 2
    result = np.sqrt((l1 + l2 + l3) / 6)
    return result.reshape(flow.shape[1], flow.shape[2], flow.shape[3])


def get_lambda2_vec(flow, dx, dy, dz):
    n, sizex, sizey, sizez = flow.shape
    Jac = get_jacobian(flow, dx, dy, dz)
    result = np.zeros((sizex, sizey, sizez))
    Jac_current = Jac.reshape(3, 3, -1).transpose(2, 0, 1)
    S = (Jac_current + Jac_current.transpose(0, 2, 1)) / 2
    Omega = (Jac_current - Jac_current.transpose(0, 2, 1)) / 2
    M = S @ S + Omega @ Omega
    eigenvals = np.linalg.eigvals(M)
    sorted_vals = np.sort(eigenvals, axis=1)
    result = sorted_vals[:, 1]
    return result.reshape(sizex, sizey, sizez)
