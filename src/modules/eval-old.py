import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import h5py
import tqdm
from maxpy.utils import *
from maxpy.algorithms import *
from scipy.ndimage import gaussian_filter
import cmasher

CMAP = plt.get_cmap("cmr.lavender")


def get_ordered_list_of_times(timesteps):
    # Get ordered list of times
    times = np.zeros(len(timesteps), dtype="int")
    for i, step in enumerate(timesteps):
        if "Timestep" not in step:  # == "config":
            continue
        times[i] = int(step.split("Timestep_")[-1])
    return np.sort(times)


def h5_to_vtk(field_name):
    pass


def load_plasma(
    filepath,
    savepath="./",
    name="output",
    vtk=False,
    plot=False,
    filter=False,
    tcrop=1,
    xcrop=1,
    sigma=10,
):
    f = h5py.File(filepath, "r")

    timesteps = list(f.keys())

    times = get_ordered_list_of_times(timesteps)[::tcrop]

    # Iterate over all time steps
    for i in tqdm.tqdm(range(len(times))):
        timestep = f"Timestep_{times[i]}"

        try:
            rhoE = np.array(f[timestep]["rho"]["rhoE[0]"])[::xcrop, ::xcrop, ::xcrop]
            rhoI = np.array(f[timestep]["rho"]["rhoI[0]"])[::xcrop, ::xcrop, ::xcrop]

            E = np.array(
                [
                    f[timestep]["felder"]["E"]["E[0]"],
                    f[timestep]["felder"]["E"]["E[1]"],
                    f[timestep]["felder"]["E"]["E[2]"],
                ]
            )[:, ::xcrop, ::xcrop, ::xcrop]

            B = np.array(
                [
                    f[timestep]["felder"]["B"]["B[0]"],
                    f[timestep]["felder"]["B"]["B[1]"],
                    f[timestep]["felder"]["B"]["B[2]"],
                ]
            )[:, ::xcrop, ::xcrop, ::xcrop]

            jL = np.array(
                [
                    f[timestep]["rho"]["rhoE[1]"],
                    f[timestep]["rho"]["rhoE[2]"],
                    f[timestep]["rho"]["rhoE[3]"],
                ]
            )[:, ::xcrop, ::xcrop, ::xcrop]

            jM = np.array(
                [
                    f[timestep]["rho"]["rhoI[1]"],
                    f[timestep]["rho"]["rhoI[2]"],
                    f[timestep]["rho"]["rhoI[3]"],
                ]
            )[:, ::xcrop, ::xcrop, ::xcrop]

            # Normalization
            # print(np.std(B, axis=(1, 2, 3)).shape)
            B = B / np.std(B)
            E = E / np.std(E)
            # jM = jM / np.std(jM, axis=(1, 2, 3))
            # jL = jM / np.std(jM, axis=(1, 2, 3))

            # smoothing
            if filter:
                E[0] = gaussian_filter(E[0], sigma=sigma)
                E[1] = gaussian_filter(E[1], sigma=sigma)
                E[2] = gaussian_filter(E[2], sigma=sigma)

                B[0] = gaussian_filter(B[0], sigma=sigma)
                B[1] = gaussian_filter(B[1], sigma=sigma)
                B[2] = gaussian_filter(B[2], sigma=sigma)

                jL[0] = gaussian_filter(jL[0], sigma=sigma)
                jL[1] = gaussian_filter(jL[1], sigma=sigma)
                jL[2] = gaussian_filter(jL[2], sigma=sigma)

                jM[0] = gaussian_filter(jM[0], sigma=sigma)
                jM[1] = gaussian_filter(jM[1], sigma=sigma)
                jM[2] = gaussian_filter(jM[2], sigma=sigma)

                rhoE = gaussian_filter(rhoE, sigma=sigma)
                rhoI = gaussian_filter(rhoI, sigma=sigma)

            linx = np.arange(0, E.shape[1])
            liny = np.arange(0, E.shape[2])
            linz = np.arange(0, E.shape[3])

            Jac_E = get_jacobian(E, 1, 1, 1)
            Jac_B = get_jacobian(B, 1, 1, 1)

            E_reshaped = E.reshape(E.shape[0], -1)
            Jac_E_reshaped = Jac_E.reshape(Jac_E.shape[0], Jac_E.shape[1], -1)

            Jv_E = np.einsum("ijk,jk->ik", Jac_E_reshaped, E_reshaped)

            B_reshaped = B.reshape(B.shape[0], -1)
            Jac_B_reshaped = Jac_B.reshape(Jac_B.shape[0], Jac_B.shape[1], -1)

            Jv_B = np.einsum("ijk,jk->ik", Jac_B_reshaped, B_reshaped)

            # shearB = get_shear_layer(B, Jac_B, linx, liny, linz)

            if vtk:
                save_as_vti(
                    E,
                    Jv_E.reshape(3, E.shape[1], E.shape[2], E.shape[3]),
                    savepath=savepath,
                    name="E_" + name + str(i),
                )
                save_as_vti(
                    B,
                    Jv_B.reshape(3, B.shape[1], B.shape[2], B.shape[3]),
                    savepath=savepath,
                    name="B_" + name + str(i),
                )
                save_as_vti(rhoE, savepath=savepath, name="rhoE_" + name + str(i))
                save_as_vti(rhoI, savepath=savepath, name="rhoI_" + name + str(i))
                save_as_vti(jL, savepath=savepath, name="jL_" + name + str(i))
                save_as_vti(jM, savepath=savepath, name="jM_" + name + str(i))
                # save_as_vti(shearB, savepath=savepath, name="shearB_" + name + str(i))

            if plot:
                fig, ax = plt.subplots(2, 2)
                ax[0, 0].imshow(np.linalg.norm(E[:, :, :, E.shape[-1] // 2], axis=0), cmap=CMAP)
                ax[0, 1].imshow(np.linalg.norm(B[:, :, :, B.shape[-1] // 2], axis=0), cmap=CMAP)
                ax[1, 0].imshow(rhoE[0, :, :, rhoE.shape[-1] // 2], cmap=CMAP)
                ax[1, 1].imshow(rhoI[0, :, :, rhoI.shape[-1] // 2], cmap=CMAP)
                plt.tight_layout()
                plt.show()
        except Exception as ex:
            print(f"Step {i} with key {times[i]} failed:", ex)
