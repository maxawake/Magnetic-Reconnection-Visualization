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

FIELD_LIST = [
    ["felder", "E", "E"],
    ["felder", "B", "B"],
    ["rho", "rhoL"],
    # ["rho", "rhoM"],
]


def get_ordered_list_of_times(timesteps):
    """Get ordered list of time steps from dings.h5

    Parameters
    ----------
    timesteps : List
        List of timesteps of type string

    Returns
    -------
    np.array
        Sorted list of timestep strings
    """
    times = np.zeros(len(timesteps), dtype="int")
    for i, step in enumerate(timesteps):
        if "Timestep" not in step:
            continue
        times[i] = int(step.split("Timestep_")[-1])
    return np.sort(times)


def nd_Gaussian_filter(vec, sigma):
    """Gaussian filter for vector vector

    Parameters
    ----------
    vec : np.array
        Vector vector of shape (3, nx, ny, nz)
    sigma : float
        Parameter of gaussian distribution

    Returns
    -------
    np.array
        Filtered vector vector
    """
    vec_filtered = np.zeros(vec.shape)
    vec_filtered[0] = gaussian_filter(vec[0], sigma=sigma)
    vec_filtered[1] = gaussian_filter(vec[1], sigma=sigma)
    vec_filtered[2] = gaussian_filter(vec[2], sigma=sigma)
    return vec_filtered


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
    tstop=None,
    resolution=(512, 1024, 256),
):
    f = h5py.File(filepath, "r")

    timesteps = list(f.keys())

    times = get_ordered_list_of_times(timesteps)[::tcrop]

    if xcrop > 1:
        resolution = (resolution[0] // xcrop, resolution[1] // xcrop, resolution[2] // xcrop)

    # grid = pv.ImageData(dimensions=dset.shape, origin=origin, spacing=spacing)

    # Iterate over all time steps
    for i in tqdm.tqdm(range(1, len(times))):
        if tstop == i:
            print(f"Break at point {tstop}")
            break
        timestep = f"Timestep_{times[i]}"
        # Iterate over all fields

        field_numpy = []
        for field_name in FIELD_LIST:
            # with Path() as p:
            try:
                # Get current vector
                if len(field_name) == 3:
                    field_h5 = f[timestep][field_name[0]][field_name[1]]

                    # Convert to numpy array and crop
                    vector = np.array(
                        [
                            field_h5[field_name[-1] + "[0]"],
                            field_h5[field_name[-1] + "[1]"],
                            field_h5[field_name[-1] + "[2]"],
                        ]
                    )[:, ::xcrop, ::xcrop, ::xcrop]

                    scalar = None

                if len(field_name) == 2:
                    field_h5 = f[timestep][field_name[0]]

                    # scalar = np.array([field_h5[field_name[-1] + "[0]"]])[0, ::xcrop, ::xcrop, ::xcrop]

                    # Convert to numpy array and crop
                    vector = np.array(
                        [
                            field_h5[field_name[-1] + "[1]"],
                            field_h5[field_name[-1] + "[2]"],
                            field_h5[field_name[-1] + "[3]"],
                        ]
                    )[:, ::xcrop, ::xcrop, ::xcrop]

                # Smoothing
                if filter:
                    vector = nd_Gaussian_filter(vector, sigma=sigma)

                    # if len(field_name) == 2:
                    #     scalar = gaussian_filter(scalar, sigma=sigma)

                # Normalization
                # scaling = np.std(vector)
                # scaling[scaling == 0] = 1e-10
                # if np.std(vector) != 0:
                #     vector = vector / np.std(vector)
                # if len(field_name) == 2:
                # scalar = scalar / np.std(scalar)
                if field_name[1] == "B":
                    vector = vector / 0.00267293778311

                # Used for parallel vectors operator
                # linx = np.linspace(0, 1, vector.shape[1])
                # liny = np.linspace(0, 1, vector.shape[2])
                # linz = np.linspace(0, 1, vector.shape[3])
                # dirDiv = get_directional_derivative(vector, linx, liny, linz)

                # if len(field_name) == 3:
                #     cross_product = np.cross(
                #         vector.reshape(vector.shape[0], -1).T, dirDiv.reshape(dirDiv.shape[0], -1).T
                #     )
                #     scalar = np.linalg.norm(cross_product, axis=1).reshape(
                #         vector.shape[1], vector.shape[2], vector.shape[3]
                #     )

                # Save as vti
                if vtk:
                    field_numpy.append((vector, "vectors-" + field_name[-1], scalar, "scalars-" + field_name[-1]))
                    # field_numpy.append()
                    ## save_as_vti(vector, dirDiv, savepath=savepath, name=f"{name}-{field_name[-1]}-{i}")
                    # save_as_vti(scalar, savepath=savepath, name=f"{name}-scalar-{i}")

                # Plot slice in the middle of current vector
                if plot:
                    field_slice = vector[:, :, :, vector.shape[-1] // 2]
                    plt.imshow(np.linalg.norm(field_slice, axis=0), cmap=CMAP)
                    plt.show()

            except Exception as ex:
                print(f"Step {i} with key {times[i]} and vector {field_name[-1]} failed:", ex)

        if vtk:
            grid = pv.ImageData(dimensions=resolution, origin=(0, 0, 0), spacing=(1, 1, 1))
            for idx in range(len(field_numpy)):
                vector, fname, scalar, sname = field_numpy[idx]
                vecs = vector.transpose(1, 2, 3, 0)
                vecs_flat = np.array([vecs[:, :, :, i].flatten(order="F") for i in range(3)]).T
                grid.point_data.set_vectors(vecs_flat, fname)
                if scalar is not None:
                    grid.point_data.set_scalars(scalar.flatten(order="F"), sname)
            # Write the VTK file
            date = print_time()
            grid.save(savepath + f"{date}-{name}.{str(i).zfill(4)}.vti")
