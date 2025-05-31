import os
import sys
from typing import Tuple, List

import numpy as np

import paraview.simple as pv
from paraview.util.vtkAlgorithm import (
    VTKPythonAlgorithmBase,
    smproxy,
    smproperty,
    smdomain,
)
from vtkmodules.vtkCommonDataModel import vtkDataObject, vtkDataSet, vtkPolyData, vtkImageData
from vtkmodules.vtkCommonCore import vtkPoints, VTK_FLOAT
from vtkmodules.vtkFiltersCore import vtkProbeFilter, vtkResampleWithDataSet
from vtkmodules.vtkFiltersGeneral import vtkWarpVector
from vtkmodules.util import numpy_support
from vtkmodules.numpy_interface import dataset_adapter as dsa
from vtkmodules.numpy_interface.algorithms import norm
from vtkmodules.numpy_interface import algorithms as alg
from vtkmodules.vtkFiltersSources import vtkPolyLineSource
from vtkmodules.vtkCommonDataModel import vtkStaticCellLocator
from vtkmodules.vtkIOXML import vtkXMLImageDataReader
from vtkmodules.vtkFiltersGeneral import vtkGradientFilter

plugin_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
src_path = os.path.join(plugin_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from mrvis.plugins.decorators import smproperty_inputarray

from prtl.vtk.prtlVectorFieldDerivatives import prtlVectorFieldDerivatives

MU0 = 4 * np.pi * 1e-7  # Permeability of free space in T*m/A
DELTA_DEFAULT = 1.0


@smproxy.filter(label="MRVIS Reconnection Rate")
@smproperty.input(name="XLine", port_index=1)
@smdomain.datatype(dataTypes=["vtkPolyData"])
@smproperty.input(name="Grid", port_index=0)
@smdomain.datatype(dataTypes=["vtkImageData"])
class mrvisReconnectionRate(VTKPythonAlgorithmBase):
    def __init__(self):
        super().__init__(nInputPorts=2, nOutputPorts=1, outputType="vtkPolyData")
        self._array_B = None
        self._array_E = None
        self._array_rho = None
        self._delta = DELTA_DEFAULT

    @smproperty_inputarray("MagneticField", idx=0, input_name="Grid", attribute_type="Vectors")
    def SetMagneticFieldArray(self, idx, port, connection, field, name):
        self._array_B = name
        self.Modified()

    @smproperty_inputarray("ElectricField", idx=1, input_name="Grid", attribute_type="Vectors")
    def SetElectricFieldArray(self, idx, port, connection, field, name):
        self._array_E = name
        self.Modified()

    @smproperty_inputarray("Density", idx=2, input_name="Grid", attribute_type="Scalars")
    def SetDensityArray(self, idx, port, connection, field, name):
        self._array_rho = name
        self.Modified()

    @smproperty.doublevector(name="Delta", default_values=DELTA_DEFAULT)
    def SetDelta(self, value):
        self._delta = float(value)
        self.Modified()

    def RequestDataObject(self, request, inInfo, outInfo):
        inp = vtkPolyData.GetData(inInfo[1], 0)
        if inp:
            out_pd = inp.NewInstance()
        else:
            out_pd = vtkPolyData()
        outInfo.GetInformationObject(0).Set(vtkDataObject.DATA_OBJECT(), out_pd)
        return 1

    def RequestData(self, request, inInfo, outInfo):
        # Get the input data objects
        grid_vtk = vtkImageData.GetData(inInfo[0], 0)
        xline_vtk = vtkPolyData.GetData(inInfo[1], 0)
        if grid_vtk is None or xline_vtk is None:
            return 1

        if not (self._array_B and self._array_E and self._array_rho):
            print("Please select MagneticField, ElectricField, and Density arrays.")
            return 0

        # Check if the input grid is a valid vtkImageData
        xline = dsa.WrapDataObject(xline_vtk)
        pts = xline.Points.copy()
        N = pts.shape[0]
        if N < 2:
            print("X-line must have at least two points.")
            return 0

        # Calculate the vector field derivatives using PRTL
        PRTLVectorFieldDerivatives1 = prtlVectorFieldDerivatives()
        PRTLVectorFieldDerivatives1.SetComputeAcceleration(False)
        PRTLVectorFieldDerivatives1.SetComputeEigenDecomposition(True)
        PRTLVectorFieldDerivatives1.SetComputeFeatureFlowField(False)
        PRTLVectorFieldDerivatives1.SetComputeStrainEigenDecomposition(False)
        PRTLVectorFieldDerivatives1.SetInputData(grid_vtk)
        PRTLVectorFieldDerivatives1.SetLeastSquaresDerivatives(False)
        PRTLVectorFieldDerivatives1.SetLeastSquaresRadius(2)
        PRTLVectorFieldDerivatives1.SetOutputDoublePrecision(True)
        PRTLVectorFieldDerivatives1.SetOutputStrainTensor(False)
        PRTLVectorFieldDerivatives1.SetOutputStructuredGrid(False)
        PRTLVectorFieldDerivatives1.SetInputArrayToProcess(0, 0, 0, 0, self._array_B)
        PRTLVectorFieldDerivatives1.Update()

        # Resample the derivatives onto the X-line
        resamp1 = vtkResampleWithDataSet()
        resamp1.SetSourceData(PRTLVectorFieldDerivatives1.GetOutput())
        locator1 = vtkStaticCellLocator()
        resamp1.SetCellLocatorPrototype(locator1)
        resamp1.SetInputData(xline_vtk)
        resamp1.SetCategoricalData(False)
        resamp1.SetComputeTolerance(True)
        resamp1.SetMarkBlankPointsAndCells(False)
        resamp1.SetPassFieldArrays(True)
        resamp1.SetPassPointArrays(False)
        resamp1.SetPassCellArrays(False)
        resamp1.SetSnapToCellWithClosestPoint(False)
        resamp1.Update()

        # Check if the resampled data contains the eigenvector
        pd1 = dsa.WrapDataObject(resamp1.GetOutput()).PointData
        eig_name = "RealEigenvectorMajor"
        if eig_name not in pd1.keys():
            print(f"Eigenvector '{eig_name}' not found after resampling.")
            return 0
        eig_line = pd1[eig_name]

        # Normalize the eigenvector
        norms = np.linalg.norm(eig_line, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        n_hat = eig_line / norms

        # Create a new vtkPolyData for the X-line with the eigenvector as point data
        flat_norm = n_hat.astype(np.float32).ravel()
        vtk_norm = numpy_support.numpy_to_vtk(num_array=flat_norm, deep=True, array_type=VTK_FLOAT)
        vtk_norm.SetNumberOfComponents(3)
        vtk_norm.SetName(eig_name)
        xline_vtk.GetPointData().AddArray(vtk_norm)
        xline_vtk.GetPointData().SetActiveVectors(eig_name)

        delta = self._delta

        # Warp the X-line points in the direction of the eigenvector
        warp_plus = vtkWarpVector()
        warp_plus.SetInputData(xline_vtk)
        warp_plus.SetScaleFactor(delta)
        warp_plus.SetInputArrayToProcess(0, 0, 0, vtkDataSet.FIELD_ASSOCIATION_POINTS, eig_name)
        warp_plus.Update()
        xline_plus = warp_plus.GetOutput()

        warp_minus = vtkWarpVector()
        warp_minus.SetInputData(xline_vtk)
        warp_minus.SetScaleFactor(-delta)
        warp_minus.SetInputArrayToProcess(0, 0, 0, vtkDataSet.FIELD_ASSOCIATION_POINTS, eig_name)
        warp_minus.Update()
        xline_minus = warp_minus.GetOutput()

        # Resample the B/E/rho fields onto the X-line
        resampE = vtkResampleWithDataSet()
        resampE.SetSourceData(grid_vtk)
        locatorE = vtkStaticCellLocator()
        resampE.SetCellLocatorPrototype(locatorE)
        resampE.SetInputData(xline_vtk)
        resampE.SetCategoricalData(False)
        resampE.SetComputeTolerance(True)
        resampE.SetMarkBlankPointsAndCells(False)
        resampE.SetPassFieldArrays(True)
        resampE.SetPassPointArrays(True)
        resampE.SetPassCellArrays(False)
        resampE.SetSnapToCellWithClosestPoint(False)
        resampE.Update()

        # Check if the resampled data contains B/E/rho
        pdE = dsa.WrapDataObject(resampE.GetOutput()).PointData
        if not (self._array_E in pdE.keys() and self._array_B in pdE.keys() and self._array_rho in pdE.keys()):
            print("Failed to resample B/E/rho onto X-line.")
            return 0

        # Resample the B/E/rho fields onto the warped X-lines
        e_line = pdE[self._array_E]
        b_line = pdE[self._array_B]

        # Calculate the reconnection rate
        f_vals = alg.sum(e_line * b_line, axis=1)

        # Calculate the mean E·B along the X-line
        seg_v = pts[1:] - pts[:-1]
        seg_l = norm(seg_v)

        # Calculate the mean E·B along the X-line segments
        f_mid = 0.5 * (f_vals[:-1] + f_vals[1:])
        phi = np.sum(f_mid * seg_l)
        L = np.sum(seg_l)
        if L == 0:
            print("Zero-length X-line.")
            return 0
        mean_EdotB = phi / L

        def resample_BR(poly: vtkPolyData) -> Tuple[np.ndarray, np.ndarray]:
            """Resample the B and rho fields onto the given polydata."""
            r = vtkResampleWithDataSet()
            r.SetSourceData(grid_vtk)
            loc = vtkStaticCellLocator()
            r.SetCellLocatorPrototype(loc)
            r.SetInputData(poly)
            r.SetCategoricalData(False)
            r.SetComputeTolerance(True)
            r.SetMarkBlankPointsAndCells(False)
            r.SetPassFieldArrays(True)
            r.SetPassPointArrays(True)
            r.SetPassCellArrays(False)
            r.SetSnapToCellWithClosestPoint(False)
            r.Update()

            pd2 = dsa.WrapDataObject(r.GetOutput()).PointData
            if self._array_B not in pd2.keys() or self._array_rho not in pd2.keys():
                print("Failed to resample B/rho at offset.")
                return None, None
            Bvals = pd2[self._array_B]
            rho_vals = pd2[self._array_rho]
            return Bvals, rho_vals

        # Resample the B and rho fields onto the warped X-lines
        Bp, rp = resample_BR(xline_plus)
        Bm, rm = resample_BR(xline_minus)
        if Bp is None or Bm is None:
            return 0

        Bmag_p = np.linalg.norm(Bp, axis=1)
        Bmag_m = np.linalg.norm(Bm, axis=1)
        rho_p = rp
        rho_m = rm

        # Calculate the average magnetic field and density
        B_in = 0.5 * (Bmag_p.mean() + Bmag_m.mean())
        rho_in = 0.5 * (rho_p.mean() + rho_m.mean())
        V_A = B_in / np.sqrt(MU0 * rho_in)

        # Calculate the reconnection rate
        R = mean_EdotB / (B_in * V_A)

        # Create the output vtkPolyData
        out_pd = vtkPolyData.GetData(outInfo, 0)
        out_pd.ShallowCopy(xline_vtk)

        rate_arr = numpy_support.numpy_to_vtk(
            num_array=np.asarray([R], dtype=np.float64), deep=True, array_type=VTK_FLOAT
        )
        rate_arr.SetName("RelativeReconnectionRate")
        out_pd.GetFieldData().AddArray(rate_arr)

        print(f"MRVIS Relative Reconnection Rate:  R = {R:.4e}")
        return 1
