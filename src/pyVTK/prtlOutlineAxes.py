__all__ = ["prtlOutlineAxes"]

from pyprtl.util.vtkAlgorithm import *
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkCellArray
from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtkmodules.numpy_interface import dataset_adapter as dsa
from vtkmodules.vtkFiltersCore import vtkTubeFilter
from vtkmodules.vtkCommonCore import vtkPoints
from vtk import vtkIdList


@smproxy.filter(label="PRTL Outline Axes")
@smhint_menu("prtl")
@smproperty.input(name="Input", port_index=0)
@smdomain.datatype(dataTypes=["vtkDataSet"])
class prtlOutlineAxes(VTKPythonAlgorithmBase):
    def __init__(self):
        self._array_field = 0
        self._array_name = None
        self._radius = 0.1
        self._resolution = 6
        VTKPythonAlgorithmBase.__init__(self, nInputPorts=1, nOutputPorts=3, outputType="vtkPolyData")

    @smproperty_inputarray("Vectors", attribute_type="Vectors")
    def SetInputArrayToProcess(self, idx, port, connection, field, name):
        self._array_field = field
        self._array_name = name
        self.Modified()

    @smproperty.doublevector(name="Radius", label="Radius", default_values=0.1)
    @smdomain.doublerange(min=1e-6)
    def SetRadius(self, d):
        self._radius = d
        self.Modified()

    @smproperty.intvector(name="Resolution", label="Resolution", default_values=0)
    @smdomain.intrange(min=0, max=128)
    def SetResolution(self, i):
        self._resolution = i
        self.Modified()

    def RequestData(self, request, inInfo, outInfo):
        input0 = vtkPolyData.GetData(inInfo[0], 0)
        output0 = dsa.WrapDataObject(vtkPolyData.GetData(outInfo, 0))
        output1 = dsa.WrapDataObject(vtkPolyData.GetData(outInfo, 1))
        output2 = dsa.WrapDataObject(vtkPolyData.GetData(outInfo, 2))

        # Get Cell Id List and poulate with Ids from PolyData
        cellIds = vtkIdList()
        # input0.GetCellPoints(currentCellId, cellIds)

        # Create three cubes
        cube1 = vtkPolyData()
        cube2 = vtkPolyData()
        cube3 = vtkPolyData()

        # Create points for the cubes
        points1 = vtkPoints()
        points2 = vtkPoints()
        points3 = vtkPoints()

        # Add points to the cubes
        for i in range(input0.GetNumberOfPoints()):
            point = input0.GetPoint(i)
            points1.InsertNextPoint(point[0], point[1], point[2])
            points2.InsertNextPoint(point[0], point[1], point[2])
            points3.InsertNextPoint(point[0], point[1], point[2])

        # Create cells for the cubes
        cells1 = vtkIdList()
        cells2 = vtkIdList()
        cells3 = vtkIdList()

        # Add cells to the cubes
        for i in range(input0.GetNumberOfCells()):
            cell = input0.GetCell(i)
            if cell.GetBounds()[1] - cell.GetBounds()[0] > 0:
                cells1.InsertNextId(i)
            if cell.GetBounds()[3] - cell.GetBounds()[2] > 0:
                cells2.InsertNextId(i)
            if cell.GetBounds()[5] - cell.GetBounds()[4] > 0:
                cells3.InsertNextId(i)

        # Create cell arrays for the cubes
        cellArray1 = vtkCellArray()
        cellArray2 = vtkCellArray()
        cellArray3 = vtkCellArray()

        # Add cells to the cell arrays
        for i in range(cells1.GetNumberOfIds()):
            cellId = cells1.GetId(i)
            cellArray1.InsertNextCell(input0.GetCell(cellId))
        for i in range(cells2.GetNumberOfIds()):
            cellId = cells2.GetId(i)
            cellArray2.InsertNextCell(input0.GetCell(cellId))
        for i in range(cells3.GetNumberOfIds()):
            cellId = cells3.GetId(i)
            cellArray3.InsertNextCell(input0.GetCell(cellId))

        # Set points and cells for the cubes
        cube1.SetPoints(points1)
        cube2.SetPoints(points2)
        cube3.SetPoints(points3)

        # Set cell arrays for the cubes
        cube1.SetLines(cellArray1)
        cube2.SetLines(cellArray2)
        cube3.SetLines(cellArray3)

        # Add tube filters to the polydatas
        tubeFilter1 = vtkTubeFilter()
        tubeFilter2 = vtkTubeFilter()
        tubeFilter3 = vtkTubeFilter()

        # Set the input for the tube filters
        tubeFilter1.SetInputData(cube1)
        tubeFilter2.SetInputData(cube2)
        tubeFilter3.SetInputData(cube3)

        # Set the radius for the tube filters
        tubeFilter1.SetRadius(self._radius)
        tubeFilter2.SetRadius(self._radius)
        tubeFilter3.SetRadius(self._radius)

        # Set the resolution for the tube filters
        tubeFilter1.SetNumberOfSides(self._resolution)
        tubeFilter2.SetNumberOfSides(self._resolution)
        tubeFilter3.SetNumberOfSides(self._resolution)

        # Update the tube filters
        tubeFilter1.Update()
        tubeFilter2.Update()
        tubeFilter3.Update()

        # Get the output of the tube filters
        tubeOutput1 = tubeFilter1.GetOutput()
        tubeOutput2 = tubeFilter2.GetOutput()
        tubeOutput3 = tubeFilter3.GetOutput()

        # Set the output data
        output0.ShallowCopy(tubeOutput1)
        output1.ShallowCopy(tubeOutput2)
        output2.ShallowCopy(tubeOutput3)

        return 1
