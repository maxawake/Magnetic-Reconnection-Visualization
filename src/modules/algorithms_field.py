import numpy as np
from numba import njit


def get_vortex_core(field, method="SH"):
    if method=="SH":
        # TODO: has to be overworked, get scalar field and do ridge extraction later
        vortex_core = np.zeros((field.dimx, field.dimy, field.dimz))
        for i in range(field.dimx):
            for j in range(field.dimy):
                for k in range(field.dimz):
                    eigenvals, eigenvecs = np.linalg.eig(field.jac[:, :, i, j, k])
                    for eigvec in eigenvecs:
                        if (np.linalg.norm(eigvec - field.field_data[:, i, j, k])) < 1.0:
                            vortex_core[i, j, k] = 1

        if np.any(vortex_core) == 1:
            print("Found points")
        else:
            print("No points found!")
        
    if method=="lambda2":
        Jac_current = field.jac.reshape(3,3,-1).transpose(2,0,1)
        S = (Jac_current + Jac_current.transpose(0,2,1)) / 2
        Omega = (Jac_current - Jac_current.transpose(0,2,1)) / 2
        M = S @ S + Omega @ Omega
        eigenvals = np.linalg.eigvals(M)
        sorted_vals = np.sort(eigenvals,axis=1)
        vortex_core = sorted_vals[:,1].reshape(field.dimx, field.dimy, field.dimz)

    if method=="parallel":
        # Reshape vec for vectorized operations
        vec_reshaped = field.field_data.reshape(field.field_data.shape[0], -1)  
        Jac_reshaped = field.jac.reshape(field.jac.shape[0], field.jac.shape[1], -1)  

        Jv = np.einsum("ijk,jk->ik", Jac_reshaped, vec_reshaped)

        parallel = np.cross(vec_reshaped.T, Jv.T)
        vortex_core = np.linalg.norm(parallel, axis=1).reshape(field.dimx, field.dimy, field.dimz)
    
    return vortex_core

@njit
def get_shear_layer(field):
    shear_layer = np.zeros((field.dimx,field.dimy,field.dimz))
    for i in range(field.dimx):
        for j in range(field.dimy):
            for k in range(field.dimz):
                Jac_current = field.jac[:,:,i,j,k]
                S = (Jac_current + Jac_current.T)/2
                eigenvals = np.linalg.eigvals(S)
                sorted_vals = np.sqrt(((eigenvals[0]-eigenvals[1])**2+(eigenvals[0]-eigenvals[2])**2+(eigenvals[1]-eigenvals[2])**2)/6)#np.sort(eigenvals)
                shear_layer[i,j,k] = sorted_vals
    return shear_layer