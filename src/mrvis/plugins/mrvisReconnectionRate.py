import os
import sys
from typing import Tuple, List

import numpy as np

# ParaView / VTK imports ------------------------------------------------------
import paraview.simple as pv
from paraview.util.vtkAlgorithm import (
    VTKPythonAlgorithmBase,
    smproxy,
    smproperty,
    smdomain,
)
from vtkmodules.vtkCommonDataModel import vtkDataObject, vtkDataSet, vtkPolyData, vtkImageData
from vtkmodules.vtkCommonCore import vtkPoints, VTK_FLOAT
from vtkmodules.vtkFiltersCore import vtkProbeFilter
from vtkmodules.numpy_interface import dataset_adapter as dsa
from vtkmodules.numpy_interface import numpy_support
from vtkmodules.numpy_interface.algorithms import norm
from vtkmodules.numpy_interface import algorithms as alg
from vtkmodules.vtkFiltersGeneral import vtkGradientFilter

plugin_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
src_path = os.path.join(plugin_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# decorator helper that adds ParaView dropdowns for array selection
from mrvis.plugins.decorators import smproperty_inputarray


MU0 = 4.0 * np.pi * 1e-7  # vacuum permeability (SI)
DELTA_DEFAULT = 5.0  # default displacement |δ| [simulation units]


@smproxy.filter(label="MRVIS Reconnection Rate")
@smproperty.input(name="Input2", port_index=1)
@smdomain.datatype(dataTypes=["vtkDataSet"])
@smproperty.input(name="Input1", port_index=0)
@smdomain.datatype(dataTypes=["vtkDataSet"])
class mrvisReconnectionRate(VTKPythonAlgorithmBase):
    def __init__(self):
        VTKPythonAlgorithmBase.__init__(self, nInputPorts=2, nOutputPorts=2, outputType="vtkPolyData")
        self._array_field = [0] * 4
        self._array_name = [None] * 4
        self._delta = DELTA_DEFAULT

    @smproperty_inputarray("Magnetic Field", idx=0, input_name="Input1", attribute_type="Vectors")
    def SetInputArrayToProcessA(self, idx, port, connection, field, name):
        self._array_field[0] = field
        self._array_name[0] = name
        self.Modified()

    @smproperty_inputarray("Electric Field", idx=1, input_name="Input1", attribute_type="Vectors")
    def SetInputArrayToProcessC(self, idx, port, connection, field, name):
        self._array_field[1] = field
        self._array_name[1] = name
        self.Modified()

    @smproperty_inputarray("Density", idx=2, input_name="Input1", attribute_type="Scalars")
    def SetInputArrayToProcessD(self, idx, port, connection, field, name):
        self._array_field[2] = field
        self._array_name[2] = name
        self.Modified()

    @smproperty_inputarray("Eigenvector", idx=3, input_name="Input1", attribute_type="Vectors")
    def SetInputArrayToProcessE(self, idx, port, connection, field, name):
        self._array_field[3] = field
        self._array_name[3] = name
        self.Modified()

    # --- SHIFT DISTANCE δ -----------------------------------------------
    @smproperty.doublevector(name="Delta", default_values=DELTA_DEFAULT)
    def SetDelta(self, value):
        self._delta = float(value)
        self.Modified()

    # --------------------------------------------------------------------
    #  Ensure we always create a vtkPolyData output, even if the user has
    #  not yet connected both inputs. Prevents REQUEST_DATA_OBJECT errors.
    # --------------------------------------------------------------------
    def RequestDataObject(self, request, inInfo, outInfo):
        inp_pd = vtkPolyData.GetData(inInfo[1], 0)
        if inp_pd:
            template = inp_pd.NewInstance()
        else:
            template = vtkPolyData()
        outInfo.GetInformationObject(0).Set(vtkDataObject.DATA_OBJECT(), template)
        return 1

    # --------------------------------------------------------------------
    #  Main computation: probe grid fields, compute n_hat from eigenvectors,
    #  integrate E along X-line, sample B and rho at ±δ, compute R.
    # --------------------------------------------------------------------
    def RequestData(self, request, inInfo, outInfo):
        """
        1) Interpolate (probe) E, B, rho, and eigenvector from the 3D grid onto
           the exact X-line points (one vtkProbeFilter call).
        2) Integrate E·t_hat along the X-line.
        3) Form ±δ offsets along the interpolated eigenvector (n_hat),
           then probe B and rho at those offset points.
        4) Compute B_in and rho_in, form V_A and the final R, and write
           it into FieldData on the output poly‐line.
        """

        # ---------------------------------------------
        #  Step 0: grab the two inputs from inInfo[]
        #   - grid_vtk  = 3D point‐mesh (vtkImageData) holding E,B,rho,eigenvector
        #   - xline_vtk = vtkPolyData of your X‐line points (XYZ coords only)
        # ---------------------------------------------
        grid_vtk = vtkImageData.GetData(inInfo[0], 0)  # port‐0
        xline_vtk = vtkPolyData.GetData(inInfo[1], 0)  # port‐1

        # If either input is missing, bail out gracefully
        if grid_vtk is None or xline_vtk is None:
            return 1

        # Wrap them so we can call xline.Points (NumPy array) and grid.PointData[key]
        grid = dsa.WrapDataObject(grid_vtk)
        xline = dsa.WrapDataObject(xline_vtk)

        # The four array‐names chosen via drop‐downs on the GUI:
        arrB_name = self._array_name[0]  # e.g. "B"
        arrE_name = self._array_name[1]  # e.g. "E"
        arrRho_name = self._array_name[2]  # e.g. "rho_Ion"
        arrEig_name = self._array_name[3]  # e.g. "RealEigenvectorMajor"

        # Make sure the user has actually selected all four arrays
        if not arrB_name or not arrE_name or not arrRho_name or not arrEig_name:
            print("Please select B, E, rho, and Eigenvector arrays on the grid.")
            return 0

        # ---------------------------------------------
        #  Step 1: Build a vtkPolyData of only the X-line points (no arrays yet)
        # ---------------------------------------------
        pts = xline.Points.copy()  # shape = (N, 3)
        N = pts.shape[0]
        if N < 2:
            print("X-line must have at least two points.")
            return 0

        # Flatten the (N,3) float32 array to 1D, then convert to vtkDataArray
        flat_xyz = pts.astype(np.float32).ravel()
        vtk_xyz = numpy_support.numpy_to_vtk(num_array=flat_xyz, deep=True, array_type=VTK_FLOAT)
        vtk_xyz.SetNumberOfComponents(3)
        tmp_pts = vtkPoints()
        tmp_pts.SetData(vtk_xyz)

        tmp_pd = vtkPolyData()
        tmp_pd.SetPoints(tmp_pts)
        # At this point tmp_pd has vertices = {xline coords}, but no point‐data arrays.

        # ---------------------------------------------
        #  Step 2: Single vtkProbeFilter to interpolate all 4 arrays onto tmp_pd
        #     We request E, B, rho, and the eigenvector.
        # ---------------------------------------------
        probeAll = vtkProbeFilter()
        probeAll.SetInputData(tmp_pd)
        probeAll.SetSourceData(grid_vtk)
        # We call SetInputArrayToProcess four times (once per array)
        probeAll.SetInputArrayToProcess(0, 0, 0, vtkDataSet.FIELD_ASSOCIATION_POINTS, arrE_name)
        probeAll.SetInputArrayToProcess(1, 0, 0, vtkDataSet.FIELD_ASSOCIATION_POINTS, arrB_name)
        probeAll.SetInputArrayToProcess(2, 0, 0, vtkDataSet.FIELD_ASSOCIATION_POINTS, arrRho_name)
        probeAll.SetInputArrayToProcess(3, 0, 0, vtkDataSet.FIELD_ASSOCIATION_POINTS, arrEig_name)
        probeAll.Update()

        # Now probeAll.GetOutput() is a vtkPolyData whose .Points == xline coords, AND
        # whose PointData contains exactly four new arrays:
        #    - arrE_name (E_interp), arrB_name (B_interp),
        #    - arrRho_name (rho_interp), arrEig_name (eig_interp).
        probed = dsa.WrapDataObject(probeAll.GetOutput()).PointData

        # Grab them as NumPy arrays of shape (N,3) or (N,)
        if arrE_name not in probed.keys():
            print(f"Electric field '{arrE_name}' not found on grid (probe failed).")
            return 0
        e_vec = probed[arrE_name]  # (N,3)

        if arrB_name not in probed.keys():
            print(f"Magnetic field '{arrB_name}' not found on grid (probe failed).")
            return 0
        B_line = probed[arrB_name]  # (N,3)

        if arrRho_name not in probed.keys():
            print(f"Density '{arrRho_name}' not found on grid (probe failed).")
            return 0
        rho_line = probed[arrRho_name]  # (N,)

        if arrEig_name not in probed.keys():
            print(f"Eigenvector '{arrEig_name}' not found on grid (probe failed).")
            return 0
        eig_line = probed[arrEig_name]  # (N,3)

        # ---------------------------------------------
        #  Step 3: Normalize the interpolated eigenvector → n_hat
        # ---------------------------------------------
        n_hat = eig_line.copy()
        norms = np.linalg.norm(n_hat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        n_hat = n_hat / norms  # shape = (N,3)

        # ---------------------------------------------
        #  Step 4: Integrate E · t_hat along the X-line
        #    We already have e_vec(i) for each point i.
        # ---------------------------------------------
        seg_v = pts[1:] - pts[:-1]  # (N-1, 3)
        seg_l = norm(seg_v)  # (N-1,)
        t_hat = seg_v / seg_l[:, None]  # (N-1, 3)
        e_mid = 0.5 * (e_vec[:-1] + e_vec[1:])  # (N-1, 3)
        e_comp = alg.sum(e_mid * t_hat, axis=1)  # (N-1,)
        phi = np.sum(e_comp * seg_l)  # scalar integral ∫ E·dl
        L = np.sum(seg_l)
        if L == 0:
            print("Zero-length X-line.")
            return 0
        e_bar = phi / L  # mean E over the length

        # ---------------------------------------------
        #  Step 5: Form ±δ offsets along n_hat, then probe B & rho there
        # ---------------------------------------------
        delta = getattr(self, "_delta", DELTA_DEFAULT)
        p_plus = pts + delta * n_hat  # (N,3)
        p_minus = pts - delta * n_hat  # (N,3)

        # A small helper that probes exactly arrB_name & arrRho_name at an Nx3 array of points
        def probe_BR(coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            flat = coords.astype(np.float32).ravel()
            vtk_arr = numpy_support.numpy_to_vtk(num_array=flat, deep=True, array_type=VTK_FLOAT)
            vtk_arr.SetNumberOfComponents(3)
            pts_obj = vtkPoints()
            pts_obj.SetData(vtk_arr)
            pd_pts = vtkPolyData()
            pd_pts.SetPoints(pts_obj)

            # Probe B
            prB = vtkProbeFilter()
            prB.SetInputData(pd_pts)
            prB.SetSourceData(grid_vtk)
            prB.SetInputArrayToProcess(0, 0, 0, vtkDataSet.FIELD_ASSOCIATION_POINTS, arrB_name)
            prB.Update()
            pdB = dsa.WrapDataObject(prB.GetOutput()).PointData
            Bvals = pdB[arrB_name]  # shape (N,3)

            # Probe rho
            prR = vtkProbeFilter()
            prR.SetInputData(pd_pts)
            prR.SetSourceData(grid_vtk)
            prR.SetInputArrayToProcess(0, 0, 0, vtkDataSet.FIELD_ASSOCIATION_POINTS, arrRho_name)
            prR.Update()
            pdR = dsa.WrapDataObject(prR.GetOutput()).PointData
            Rvals = pdR[arrRho_name]  # shape (N,)

            return Bvals, Rvals

        Bp, rp = probe_BR(p_plus)
        Bm, rm = probe_BR(p_minus)

        mask_p = np.isfinite(rp)
        mask_m = np.isfinite(rm)
        if not mask_p.any() or not mask_m.any():
            print("Probed points outside grid – adjust Delta.")
            return 0

        Bmag_p = np.linalg.norm(Bp[mask_p], axis=1)  # (n_valid_p,)
        Bmag_m = np.linalg.norm(Bm[mask_m], axis=1)  # (n_valid_m,)
        rho_p = rp[mask_p]  # (n_valid_p,)
        rho_m = rm[mask_m]  # (n_valid_m,)

        B_in = 0.5 * (Bmag_p.mean() + Bmag_m.mean())
        rho_in = 0.5 * (rho_p.mean() + rho_m.mean())
        V_A = B_in / np.sqrt(MU0 * rho_in)

        # ---------------------------------------------
        #  Step 6: Compute the relative reconnection rate R = e_bar / (B_in V_A)
        # ---------------------------------------------
        R = e_bar / (B_in * V_A)

        # ---------------------------------------------
        #  Step 7: Write the output poly‐line and attach R to FieldData
        # ---------------------------------------------
        out_pd = vtkPolyData.GetData(outInfo, 0)
        out_pd.ShallowCopy(xline_vtk)

        rate_arr = numpy_support.numpy_to_vtk(num_array=np.asarray([R], dtype=np.float64), deep=True)
        rate_arr.SetName("RelativeReconnectionRate")
        out_pd.GetFieldData().AddArray(rate_arr)

        print(f"MRVIS Relative Reconnection Rate:  R = {R:.4e}")
        return 1
