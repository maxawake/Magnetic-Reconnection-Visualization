import os
import sys
from typing import List, Tuple
import tqdm

import numpy as np
import paraview.simple as pv
from paraview.util.vtkAlgorithm import (
    VTKPythonAlgorithmBase,
    smdomain,
    smproperty,
    smproxy,
)
from vtk import vtkIdList
from vtkmodules.numpy_interface import algorithms as alg
from vtkmodules.numpy_interface import dataset_adapter as dsa
from vtkmodules.numpy_interface.algorithms import norm
from vtkmodules.util import numpy_support
from vtkmodules.vtkCommonCore import VTK_FLOAT, vtkPoints
from vtkmodules.vtkCommonDataModel import (
    vtkDataObject,
    vtkDataSet,
    vtkImageData,
    vtkPolyData,
    vtkStaticCellLocator,
    vtkCellArray,
)
from vtkmodules.vtkFiltersCore import vtkResampleWithDataSet, vtkCleanPolyData
from vtkmodules.vtkFiltersGeneral import vtkWarpVector


plugin_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
src_path = os.path.join(plugin_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from prtl.vtk.prtlVectorFieldDerivatives import prtlVectorFieldDerivatives

from mrvis.plugins.decorators import smproperty_inputarray, smdomain_boolean

DELTA_DEFAULT = 1.0
SPEED_OF_LIGHT = 3e10  # cm/s


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
        self._clean_input = 0
        self._verbose = 0
        self._delta = DELTA_DEFAULT
        self._save = 0
        self._save_path = "./"

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
    @smdomain.doublerange(min=0.0, max=10.0, step=0.1)
    def SetDelta(self, value):
        self._delta = float(value)
        self.Modified()

    @smproperty.intvector(name="CleanInput", label="Clean Input", default_values=0)
    @smdomain_boolean()
    def SetCleanInput(self, value):
        self._clean_input = value
        self.Modified()

    @smproperty.intvector(name="Verbose", label="Verbose", default_values=0)
    @smdomain_boolean()
    def SetVerbose(self, value):
        self._verbose = value
        self.Modified()

    @smproperty.intvector(name="SaveResults", label="Save Results", default_values=0)
    @smdomain_boolean()
    def SetSaveResult(self, value):
        """Set the save path for reconnection rates."""
        self._save = value
        self.Modified()

    @smproperty.stringvector(name="SavePath", label="Save Path")
    def SetSavePath(self, path):
        """Set the save path for reconnection rates."""
        self._save_path = path
        self.Modified()

    def RequestDataObject(self, request, inInfo, outInfo):
        inp = vtkPolyData.GetData(inInfo[1], 0)
        if inp:
            out_pd = inp.NewInstance()
        else:
            out_pd = vtkPolyData()
        outInfo.GetInformationObject(0).Set(vtkDataObject.DATA_OBJECT(), out_pd)
        return 1

    def compute_reconnection_rate_for_line(self, grid_vtk, xline_vtk, derivs) -> Tuple[float, List[int]]:
        xline = dsa.WrapDataObject(xline_vtk)
        pts = xline.Points.copy()
        N = pts.shape[0]
        if N < 2:
            if self._verbose:
                print("Warning: Line segment has fewer than 2 points, cannot compute reconnection rate.")
            return 0.0

        resamp1 = vtkResampleWithDataSet()
        resamp1.SetSourceData(derivs.GetOutput())
        resamp1.SetCellLocatorPrototype(vtkStaticCellLocator())
        resamp1.SetInputData(xline_vtk)
        resamp1.SetPassFieldArrays(True)
        resamp1.SetPassPointArrays(False)
        resamp1.SetCategoricalData(False)
        resamp1.Update()

        pd1 = dsa.WrapDataObject(resamp1.GetOutput()).PointData
        eig_name = "RealEigenvectorMajor"
        if eig_name not in pd1.keys():
            if self._verbose:
                print(f"Warning: Eigenvector '{eig_name}' not found in point data.")
            return 0.0

        n_hat = pd1[eig_name]
        norms = np.linalg.norm(n_hat, axis=1, keepdims=True)
        n_hat /= norms

        flat_norm = n_hat.astype(np.float32).ravel()
        vtk_norm = numpy_support.numpy_to_vtk(flat_norm, deep=True, array_type=VTK_FLOAT)
        vtk_norm.SetNumberOfComponents(3)
        vtk_norm.SetName(eig_name)
        xline_vtk.GetPointData().AddArray(vtk_norm)
        xline_vtk.GetPointData().SetActiveVectors(eig_name)

        delta = self._delta
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

        def resample_fields(poly):
            resampler = vtkResampleWithDataSet()
            resampler.SetSourceData(grid_vtk)
            resampler.SetCellLocatorPrototype(vtkStaticCellLocator())
            resampler.SetInputData(poly)
            resampler.SetPassFieldArrays(True)
            resampler.SetPassPointArrays(True)
            resampler.SetCategoricalData(False)
            resampler.Update()
            return dsa.WrapDataObject(resampler.GetOutput()).PointData

        point_data = resample_fields(xline_vtk)
        B_line = point_data[self._array_B]
        E_line = point_data[self._array_E]
        if B_line is None or E_line is None:
            Exception("Magnetic or Electric field data is missing.")

        # Normalize B
        Bmag = np.linalg.norm(B_line, axis=1)
        Bmag[Bmag == 0] = 1.0  # Avoid division by zero

        f_vals = alg.sum(E_line * B_line, axis=1)
        f_vals /= Bmag

        seg_l = norm(pts[1:] - pts[:-1])
        f_mid = 0.5 * (f_vals[:-1] + f_vals[1:])
        phi = np.sum(f_mid * seg_l)
        L = np.sum(seg_l)
        if L == 0:
            if self._verbose:
                print("Warning: Length of line segment is zero, cannot compute reconnection rate.")
            return 0.0

        mean_EdotB = np.abs(phi / L)
        point_data_plus = resample_fields(xline_plus)
        point_data_minus = resample_fields(xline_minus)
        Bp = point_data_plus[self._array_B]
        Bm = point_data_minus[self._array_B]
        rp = point_data_plus[self._array_rho]
        rm = point_data_minus[self._array_rho]

        if Bp is None or Bm is None or rp is None or rm is None:
            Exception("Magnetic field or density data is missing.")

        Bmag_p = np.linalg.norm(Bp, axis=1)
        Bmag_m = np.linalg.norm(Bm, axis=1)
        rho_in = 0.5 * (np.mean(rp) + np.mean(rm))
        B_in = 0.5 * (np.mean(Bmag_p) + np.mean(Bmag_m))
        epsilon = 1e-18  # or choose a value appropriate for your density scale
        if rho_in < epsilon:
            if self._verbose:
                print(f"Warning: Density too small ({rho_in:.2e}), clamping to {epsilon}.")
            rho_in = epsilon
        V_A = B_in / np.sqrt(4 * np.pi * rho_in)
        R = mean_EdotB * SPEED_OF_LIGHT / (B_in * V_A)

        if self._verbose:
            print(" --- Reconnection Rate Computation ---")
            print("Average Warp Δ:", np.sqrt(np.sum((pts - pts + delta * n_hat) ** 2, axis=1)).min())
            print("B mean:", np.mean(B_line, axis=0))
            print("E mean:", np.mean(E_line, axis=0))
            print("Mean E·B:", mean_EdotB)
            print("f mean:", np.mean(f_vals), "f mid mean:", np.mean(f_mid))
            print("Line segment length L:", L)
            print("Eigenvector norm stats:", np.min(norms), np.max(norms))
            print(
                "rho plus:",
                np.mean(point_data_plus[self._array_rho]),
                "rho minus",
                np.mean(point_data_minus[self._array_rho]),
            )
            print("Bp mean:", np.mean(Bp, axis=0), "Bp norm:", np.linalg.norm(Bp, axis=1).mean())
            print("Bm mean:", np.mean(Bm, axis=0), "Bm norm:", np.linalg.norm(Bm, axis=1).mean())
            print("Alfvenic speed V_A:", V_A)
            print(f"rho_in: {rho_in}, B_in: {B_in}")
            print("Reconnection rate R:", R)

        return R

    def RequestData(self, request, inInfo, outInfo):
        grid_vtk = vtkImageData.GetData(inInfo[0], 0)
        xlines_all = vtkPolyData.GetData(inInfo[1], 0)
        out_pd = vtkPolyData.GetData(outInfo, 0)
        out_pd.ShallowCopy(xlines_all)
        pts = dsa.WrapDataObject(xlines_all).Points
        if self._verbose:
            print("Point extent:", np.min(pts, axis=0), "to", np.max(pts, axis=0))

        if self._clean_input:
            # create a new vtkCleanPolyData
            Clean1 = vtkCleanPolyData()
            Clean1.SetAbsoluteTolerance(1.0)
            Clean1.SetConvertLinesToPoints(True)
            Clean1.SetConvertPolysToLines(True)
            Clean1.SetConvertStripsToPolys(True)
            Clean1.SetInputData(xlines_all)
            Clean1.SetPieceInvariant(True)
            Clean1.SetPointMerging(True)
            Clean1.SetTolerance(0.0)
            Clean1.SetToleranceIsAbsolute(False)
            Clean1.Update()
            xlines_all = Clean1.GetOutput()

        derivs = prtlVectorFieldDerivatives()
        derivs.SetComputeAcceleration(False)
        derivs.SetComputeEigenDecomposition(True)
        derivs.SetComputeFeatureFlowField(False)
        derivs.SetComputeStrainEigenDecomposition(False)
        derivs.SetInputData(grid_vtk)
        derivs.SetLeastSquaresDerivatives(False)
        derivs.SetLeastSquaresRadius(2)
        derivs.SetOutputDoublePrecision(True)
        derivs.SetOutputStrainTensor(False)
        derivs.SetOutputStructuredGrid(False)
        derivs.SetInputArrayToProcess(0, 0, 0, 0, self._array_B)
        derivs.Update()

        if not all([grid_vtk, xlines_all]):
            return 1
        if not all([self._array_B, self._array_E, self._array_rho]):
            if self._verbose:
                print("Warning: Magnetic field, electric field, or density array not set.")
            return 0

        num_cells = xlines_all.GetNumberOfCells()
        rates = np.full(xlines_all.GetNumberOfPoints(), 0.0)

        for cell_id in tqdm.tqdm(range(num_cells)):
            id_list = vtkIdList()
            xlines_all.GetCellPoints(cell_id, id_list)
            point_ids = [id_list.GetId(i) for i in range(id_list.GetNumberOfIds())]

            if len(point_ids) < 2:
                continue

            # Reconstruct line segment as new vtkPolyData
            subline_pts = [xlines_all.GetPoint(pid) for pid in point_ids]
            points = vtkPoints()
            for pt in subline_pts:
                points.InsertNextPoint(pt)

            lines = vtkCellArray()
            lines.InsertNextCell(len(point_ids))
            for i in range(len(point_ids)):
                lines.InsertCellPoint(i)

            subline_vtk = vtkPolyData()
            subline_vtk.SetPoints(points)
            subline_vtk.SetLines(lines)

            R = self.compute_reconnection_rate_for_line(grid_vtk, subline_vtk, derivs)
            if np.isnan(R) or np.isinf(R) or R < 0:
                if self._verbose:
                    print(f"Warning: Reconnection rate for cell {cell_id} is NaN, skipping.")
                R = 0.0

            for pid in point_ids:
                rates[pid] = R

            if self._verbose:
                print("\n")

        # Get the current time step
        if self._save:
            if not os.path.exists(self._save_path):
                os.makedirs(self._save_path)
            time_step = pv.GetAnimationScene().TimeKeeper.Time
            np.savetxt(
                os.path.join(self._save_path, f"reconnection_rates_step_{str(int(time_step)).zfill(4)}.txt"), rates
            )
            if self._verbose:
                print(f"Saving reconnection rates to {self._save_path}")

        rate_array = numpy_support.numpy_to_vtk(rates, deep=True, array_type=VTK_FLOAT)
        rate_array.SetName("ReconnectionRate")
        out_pd.GetPointData().AddArray(rate_array)
        return 1
