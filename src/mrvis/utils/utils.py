import os
from datetime import datetime

import numpy as np
import pyvista as pv
import tqdm
import vtk
import vtk.util.numpy_support as numpy_support


def print_time():
    """Print current time as string for file naming

    Returns:
        String: Formatted Date
    """
    return datetime.now().strftime("%Y-%m-%d-")


def lerp(field1, field2, steps=100, save=False, name="output"):
    """Linear interpolation of two fields

    Args:
        field1 (Field): Initial field
        field2 (Field): Final field
        steps (int, optional): Number of steps linear interpolation should be evaluated on. Defaults to 100.

    Returns:
        np.array: N+1 dimensional tensor containing all time steps
    """
    time_tensor = np.zeros((steps, *(field1.shape)))
    for i in tqdm.tqdm(range(steps)):
        time_tensor[i] = (1 - i / (steps - 1)) * field1 + i / (steps - 1) * field2
        if save:
            save_as_vti(time_tensor[i], "/home/max/Temp/", f"{name}{i}")
    return time_tensor


def save_as_vti(dset, grad=None, origin=(0, 0, 0), spacing=(1, 1, 1), savepath="./", name="output"):
    """Save numpy 3d data array as vtkImageData for visualization in Paraview

    Args:
        dset (np.ndarray): Data array containing the scalar or vector values
        savepath (str, optional): Path where to save the .vti file. Defaults to "./".
        name (str, optional): Name of the file and dataset. Defaults to "output".

    Raises:
        NotImplementedError: Only 3D scalar or vector fields are supported right now
    """
    # Create a PyVista grid object
    if len(dset.shape) == 4:
        # If vector field
        dset = dset.transpose(1, 2, 3, 0)
        dset_flat = np.array([dset[:, :, :, i].flatten(order="F") for i in range(3)]).T
        grid = pv.ImageData(dimensions=dset.shape, origin=origin, spacing=spacing)
        grid.point_data.set_vectors(dset_flat, "01_vectors")

        if grad is not None:
            grad = grad.transpose(1, 2, 3, 0)
            grad_flat = np.array([grad[:, :, :, i].flatten(order="F") for i in range(3)]).T
            grid.point_data.set_array(grad_flat, "02_Jac")

    elif len(dset.shape) == 3:
        # If scalar field
        grid = pv.ImageData(dimensions=dset.shape, origin=origin, spacing=spacing)
        grid.point_data.set_scalars(dset.flatten(order="F"), "field")
    else:
        raise NotImplementedError()

    # Write the VTK file
    date = print_time()
    grid.save(savepath + date + name + ".vti")


def save_as_unstructued_grid(u, x, y, z, plot=False, save=False, savepath="./", name="output"):
    """Create vtkUnstructuredGrid from numpy 3d array

    Args:
        u (np.ndarray): 3D scalar field
        x (np.ndarray): 3D x coordinates
        y (np.ndarray): 3D y coordinates
        z (np.ndarray): 3D z coordinates
        plot (bool, optional): Set to true if the grid should be plotted. Defaults to False.
        save (bool, optional): Set to true if the grid should be saved to disk. Defaults to False.
        savepath (str, optional): If save is set to true, this is the save path. Defaults to "./".
        name (str, optional): If save is set to true, this is the save name of the file.
        Note: File ending gets added automatically. Defaults to "output".

    Returns:
        vtk.UnstructuredGrid: Unstructured grid from numpy 3d array
    """
    grid = pv.StructuredGrid(x, y, z)
    grid["scalars"] = u.ravel(order="f")

    ugrid = pv.UnstructuredGrid(grid)

    if plot:
        ugrid.plot(show_edges=True)
    if save:
        ugrid.save(os.path.join(savepath, name + ".vtk"))

    return ugrid


def np_to_vtk(data):
    """
    Convert a numpy array to a vtkImageData object
    """
    data_type = vtk.VTK_FLOAT
    shape = data.shape

    flat_data_array = data.flatten()
    vtk_data = numpy_support.numpy_to_vtk(num_array=flat_data_array, deep=True, array_type=data_type)

    img = vtk.vtkImageData()
    img.GetPointData().SetScalars(vtk_data)
    img.SetDimensions(shape[0], shape[1], shape[2])
    return img


def vtk_to_np(data):
    """
    Convert a vtkImageData object to a numpy array

    Args:
        data: vtkImageData object

    Returns:
        numpy array
    """
    temp = numpy_support.vtk_to_numpy(data.GetPointData().GetScalars())
    dims = data.GetDimensions()
    component = data.GetNumberOfScalarComponents()
    if component == 1:
        numpy_data = temp.reshape(dims[2], dims[1], dims[0])
        numpy_data = numpy_data.transpose(2, 1, 0)
    elif component == 3 or component == 4:
        if dims[2] == 1:  # a 2D RGB image
            numpy_data = temp.reshape(dims[1], dims[0], component)
            numpy_data = numpy_data.transpose(0, 1, 2)
            numpy_data = np.flipud(numpy_data)
        else:
            raise RuntimeError("unknow type")
    return numpy_data
