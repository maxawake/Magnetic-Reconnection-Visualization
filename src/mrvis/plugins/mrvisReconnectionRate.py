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
from vtkmodules.vtkCommonDataModel import vtkDataObject, vtkDataSet, vtkPolyData
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkFiltersCore import vtkProbeFilter
from vtkmodules.numpy_interface import dataset_adapter as dsa
from vtkmodules.numpy_interface.algorithms import norm
from vtkmodules.numpy_interface import algorithms as alg


plugin_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
src_path = os.path.join(plugin_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# decorator helper that adds ParaView dropdowns for array selection
from mrvis.plugins.decorators import smproperty_inputarray

# -----------------------------------------------------------------------------
#  MRVIS Reconnection-Rate filter  (DROP-DOWN version)
# -----------------------------------------------------------------------------
#  Port 0 : vtkPolyData – poly-line that traces the X-line.  Must carry at least
#           the Eigen-vector array that points normal to the sheet.  All other
#           field names (E, B, rho) are **configurable from dropdown menus** so
#           the filter can be reused with any code’s naming convention.
#  Port 1 : vtkDataSet   – the full 3-D field grid, used for probing upstream
#           B and rho.
# -----------------------------------------------------------------------------

# ---- internal constants & defaults -----------------------------------------
MU0 = 4.0 * np.pi * 1e-7  # SI vacuum permeability
DELTA_DEFAULT = 5.0  # physical displacement (Δ)


# -----------------------------------------------------------------------------
@smproxy.filter(label="MRVIS Reconnection Rate (dropdown)")
# X-line poly-line input -------------------------------------------------------
# First input: keep ParaView-compatible name *Input* so smproperty_inputarray domains work
@smproperty.input(name="Input Field", port_index=0)
@smdomain.datatype(dataTypes=["vtkDataSet"])
# Field grid input ------------------------------------------------------------
@smproperty.input(name="Input Line", port_index=1)
@smdomain.datatype(dataTypes=["vtkPolyData"])
class MRVISReconnectionRate(VTKPythonAlgorithmBase):
    """Compute the *relative* reconnection rate R from an X-line poly-line.

    The user can choose which point-data arrays furnish the electric field **E**,
    magnetic field **B**, mass-density rho, and sheet-normal eigen-vector via drop-
    down menus in the GUI.
    """

    # ---------------- constructor & array bookkeeping -----------------------
    def __init__(self):
        super().__init__(nInputPorts=2, nOutputPorts=1, outputType="vtkPolyData")
        self._array_field = [0] * 4
        self._array_name = [None] * 4
        self._array_modified = [False] * 4

        self._delta = DELTA_DEFAULT  # physical displacement (Δ)

    def SetInputArrayToProcess(self, idx, port, connection, field, name):
        if self._array_name[idx] != name or self._array_field[idx] != field:
            self._array_modified[idx] = True
        self._array_field[idx] = field
        self._array_name[idx] = name
        self.Modified()

    # ---- GUI-visible array pickers -----------------------------------------
    @smproperty_inputarray(
        "ElectricField", attribute_type="Vectors", none_string="None", idx=0, command="SetInputArrayToProcess"
    )
    def SetElectricFieldArray():
        """Dropdown for **E** vector array."""
        pass

    @smproperty_inputarray(
        "MagneticField", attribute_type="Vectors", none_string="None", idx=1, command="SetInputArrayToProcess"
    )
    def SetMagneticFieldArray():
        """Dropdown for **B** vector array."""
        pass

    @smproperty_inputarray(
        "Density", attribute_type="Scalars", none_string="None", idx=2, command="SetInputArrayToProcess"
    )
    def SetDensityArray():
        """Dropdown for scalar mass-density rho array."""
        pass

    @smproperty_inputarray(
        "SheetNormal", attribute_type="Vectors", none_string="None", idx=3, command="SetInputArrayToProcess"
    )
    def SetNormalVectorArray():
        """Dropdown for sheet-normal eigen-vector array."""
        pass

    # ---- shift distance δ property ----------------------------------------
    @smproperty.doublevector(name="Delta", default_values=DELTA_DEFAULT)
    def SetDelta(self, value):
        self._delta = float(value)
        self.Modified()

    # ----------------------------------------------------------------------
    #  Standard VTK pipeline overrides
    # ----------------------------------------------------------------------

    def RequestDataObject(self, request, inInfo, outInfo):
        # ensure output type matches input polydata
        xline = vtkPolyData.GetData(inInfo[0], 0)
        if not xline:
            return 0
        out_pd = vtkDataObject.GetData(outInfo, 0)
        if not out_pd or not out_pd.IsA("vtkPolyData"):
            new_pd = xline.NewInstance()
            outInfo.GetInformationObject(0).Set(vtkDataObject.DATA_OBJECT(), new_pd)
        return 1

    # ----------------------------------------------------------------------
    def RequestData(self, request, inInfo, outInfo):
        # ------------ fetch the two inputs ---------------------------------
        xline_vtk: vtkPolyData = vtkPolyData.GetData(inInfo[0], 0)
        field_vtk: vtkDataSet = vtkDataSet.GetData(inInfo[1], 0)

        if xline_vtk is None:
            self.GetErrorObserver().ErrorMessage("X-line input missing.")
            return 0
        if field_vtk is None:
            self.GetErrorObserver().ErrorMessage("Field-grid input missing.")
            return 0

        xline = dsa.WrapDataObject(xline_vtk)

        # ---------------- verify selected arrays ---------------------------
        arrE, arrB, arrRho, arrNorm = self._arr_name
        needed = [arrE, arrB, arrRho, arrNorm]
        missing = [name for name in needed if name not in xline.PointData.keys()]
        if missing:
            self.GetErrorObserver().ErrorMessage(f"Array(s) {missing} not present on X-line input.")
            return 0

        # --------------- STEP 1: integrate E·t̂ along the line ------------
        pts = xline.Points
        if pts.shape[0] < 2:
            self.GetErrorObserver().ErrorMessage("Polyline must have ≥2 points.")
            return 0

        e_vec = xline.PointData[arrE]
        seg_vec = pts[1:] - pts[:-1]
        seg_len = norm(seg_vec)
        t_hat = seg_vec / seg_len[:, None]
        e_mid = 0.5 * (e_vec[:-1] + e_vec[1:])
        e_comp = alg.sum(e_mid * t_hat, axis=1)
        phi = np.sum(e_comp * seg_len)
        L = np.sum(seg_len)
        if L == 0:
            self.GetErrorObserver().ErrorMessage("Zero-length X-line.")
            return 0
        e_bar = phi / L

        # --------------- STEP 2: probe upstream B and ρ -------------------
        n_hat = xline.PointData[arrNorm]
        n_hat = n_hat / np.linalg.norm(n_hat, axis=1, keepdims=True)

        p_plus = pts + self._delta * n_hat
        p_minus = pts - self._delta * n_hat

        def probe(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            tmp_pts = vtkPoints()
            tmp_pts.SetData(dsa.numpyTovtk(points))
            tmp_pd = vtkPolyData()
            tmp_pd.SetPoints(tmp_pts)
            probe = vtkProbeFilter()
            probe.SetInputData(tmp_pd)
            probe.SetSourceData(field_vtk)
            probe.Update()
            probed = dsa.WrapDataObject(probe.GetOutput())
            return probed.PointData[arrB], probed.PointData[arrRho]

        B_plus, rho_plus = probe(p_plus)
        B_minus, rho_minus = probe(p_minus)

        valid_plus = np.isfinite(rho_plus)
        valid_minus = np.isfinite(rho_minus)
        if not valid_plus.any() or not valid_minus.any():
            self.GetErrorObserver().ErrorMessage("Probe points outside field grid – adjust Δ or data extent.")
            return 0

        B_in = 0.5 * (
            np.linalg.norm(B_plus[valid_plus], axis=1).mean() + np.linalg.norm(B_minus[valid_minus], axis=1).mean()
        )
        rho_in = 0.5 * (rho_plus[valid_plus].mean() + rho_minus[valid_minus].mean())

        V_A = B_in / np.sqrt(MU0 * rho_in)
        R = e_bar / (B_in * V_A)

        # --------------- output -------------------------------------------
        out_pd: vtkPolyData = vtkPolyData.GetData(outInfo, 0)
        out_pd.ShallowCopy(xline_vtk)  # preserve geometry & arrays

        rate_arr = dsa.numpyTovtk(np.array([R], dtype=np.float64))
        rate_arr.SetName("RelativeReconnectionRate")
        out_pd.GetFieldData().AddArray(rate_arr)

        print(f"MRVIS Relative Reconnection Rate: R = {R:.4e}")
        return 1
