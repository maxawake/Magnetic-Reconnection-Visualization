# ParallelVectorsImageFilter_safe.py
# ----------------------------------
# ParaView Python filter: finds points where two 3-component point-data
# vectors are parallel inside vtkImageData.
#
# Key safety features:
#   * __init__ is completely inert
#   * RequestDataObject creates an empty vtkPolyData unconditionally
#   * all sanity checks live in RequestData and simply raise RuntimeError
#   * properties have ArrayListDomain so the GUI lists only valid arrays

import vtk
import numpy as np
from paraview.util.vtkAlgorithm import VTKPythonAlgorithmBase, smproxy, smproperty, smdomain


@smproxy.filter(name="ParallelVectorsImageFilterSafe", label="Parallel Vectors (Image – safe)")
class ParallelVectorsImageFilterSafe(VTKPythonAlgorithmBase):
    def __init__(self):
        super().__init__(nInputPorts=1, nOutputPorts=1, outputType="vtkPolyData")
        self._first = ""
        self._second = ""

    # ---- UI properties -------------------------------------------------
    @smproperty.stringvector(name="FirstVectorField")
    @smdomain.xml("""
        <ArrayListDomain name="vectors" attribute_type="Vectors"
                         number_of_components="3">
          <RequiredProperties>
            <Property name="Input" function="Input" />
          </RequiredProperties>
        </ArrayListDomain>
    """)
    def SetFirst(self, name):
        self._first = name
        self.Modified()

    @smproperty.stringvector(name="SecondVectorField")
    @smdomain.xml("""
        <ArrayListDomain name="vectors" attribute_type="Vectors"
                         number_of_components="3">
          <RequiredProperties>
            <Property name="Input" function="Input" />
          </RequiredProperties>
        </ArrayListDomain>
    """)
    def SetSecond(self, name):
        self._second = name
        self.Modified()

    # ---- accept only vtkImageData -------------------------------------
    def FillInputPortInformation(self, port, info):
        info.Set(self.INPUT_REQUIRED_DATA_TYPE(), "vtkImageData")
        return 1

    # ---- always create a valid (empty) output object -------------------
    def RequestDataObject(self, request, inInfo, outInfo):
        outInfo.GetInformationObject(0).Set(vtk.vtkDataObject.DATA_OBJECT(), vtk.vtkPolyData())
        return 1

    # ---- main computation ---------------------------------------------
    def RequestData(self, request, inInfo, outInfo):  # noqa: N802
        img = vtk.vtkImageData.GetData(inInfo[0])
        poly = vtk.vtkPolyData.GetData(outInfo)

        if not self._first or not self._second:
            raise RuntimeError("Choose both FirstVectorField and SecondVectorField before pressing Apply.")

        v_arr = img.GetPointData().GetArray(self._first)
        w_arr = img.GetPointData().GetArray(self._second)
        if v_arr is None or w_arr is None:
            raise RuntimeError("Chosen arrays not found in the data set.")
        if v_arr.GetNumberOfComponents() != 3 or w_arr.GetNumberOfComponents() != 3:
            raise RuntimeError("Both arrays must have exactly 3 components.")

        # --- loop over cells -------------------------------------------
        pts = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        for cid in range(img.GetNumberOfCells()):
            cell = img.GetCell(cid)
            if cell.GetCellDimension() != 3:  # skip degenerate cells
                continue
            for tri in _triangulate(cell):
                v, w = _vectors(tri, img, v_arr, w_arr)
                st = _parallel_location(v, w)
                if st is None:
                    continue
                p = _interpolate(st, tri, img)
                pid = pts.InsertNextPoint(p)
                lines.InsertNextCell(1)
                lines.InsertCellPoint(pid)

        poly.SetPoints(pts)
        poly.SetLines(lines)
        return 1


# ------ helper functions (pure NumPy/Vtk) -------------------------------


def _triangulate(cell):
    tris = []
    for f in range(cell.GetNumberOfFaces()):
        face = cell.GetFace(f)
        n = face.GetNumberOfPoints()
        for i in range(1, n - 1):
            tris.append([face.GetPointId(0), face.GetPointId(i), face.GetPointId(i + 1)])
    return tris


def _vectors(tri, ds, v_arr, w_arr):
    v = np.array([v_arr.GetTuple(pid) for pid in tri])
    w = np.array([w_arr.GetTuple(pid) for pid in tri])
    return v, w


def _parallel_location(v_vecs, w_vecs):
    V = np.column_stack([v_vecs[1] - v_vecs[0], v_vecs[2] - v_vecs[0], v_vecs[0]])
    W = np.column_stack([w_vecs[1] - w_vecs[0], w_vecs[2] - w_vecs[0], w_vecs[0]])
    detV, detW = np.linalg.det(V), np.linalg.det(W)
    if abs(detV) < 1e-12 and abs(detW) < 1e-12:
        return None
    M = np.linalg.solve(W, V) if abs(detW) > abs(detV) else np.linalg.solve(V, W)
    vals, vecs = np.linalg.eig(M)
    for lam, vec in zip(vals, vecs.T):
        if np.iscomplex(lam) or abs(vec[2]) < 1e-12:
            continue
        vec /= vec[2]
        s, t = vec[0].real, vec[1].real
        if s >= 0 and t >= 0 and s + t <= 1:
            return s, t
    return None


def _interpolate(st, tri, ds):
    s, t = st
    bc = np.array([1 - s - t, s, t])
    xyz = np.array([ds.GetPoint(pid) for pid in tri])
    return bc @ xyz
