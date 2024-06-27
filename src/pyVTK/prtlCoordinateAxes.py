__all__ = ["prtlCoordinateAxes"]

from pyprtl.util.vtkAlgorithm import *
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkPolyLine, vtkPolyData, vtkCellArray
from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtkmodules.numpy_interface import dataset_adapter as dsa
from vtkmodules.vtkFiltersCore import vtkTubeFilter, vtkAppendPolyData
from vtkmodules.vtkFiltersSources import vtkConeSource


@smproxy.filter(label="PRTL Coordinate Axes")
@smhint_menu("prtl")
@smproperty.input(name="Input", port_index=0)
@smdomain.datatype(dataTypes=["vtkDataSet"])
class prtlCoordinateAxes(VTKPythonAlgorithmBase):
    def __init__(self):
        self._array_field = 0
        self._array_name = None
        self._resolution = 6
        self._scale = 1
        VTKPythonAlgorithmBase.__init__(self, nInputPorts=1, nOutputPorts=3, outputType="vtkPolyData")

    @smproperty.intvector(name="Resolution", label="Resolution", default_values=6)
    @smdomain.intrange(min=6, max=100)
    def SetResolution(self, i):
        self._resolution = i
        self.Modified()

    @smproperty.doublevector(name="Size", label="Size", default_values=1.0)
    @smdomain.doublerange(min=0.0, max=100.0)
    def SetSize(self, d):
        self._scale = d
        self.Modified()

    @staticmethod
    def GetPolyLine(p0, p1):
        # Create a vtkPoints object and store the points in it
        points = vtkPoints()
        points.InsertNextPoint(p0)
        points.InsertNextPoint(p1)

        polyLine = vtkPolyLine()
        polyLine.GetPointIds().SetNumberOfIds(2)
        for i in range(2):
            polyLine.GetPointIds().SetId(i, i)

        # Create a cell array to store the lines in and add the lines to it
        cells = vtkCellArray()
        cells.InsertNextCell(polyLine)

        # Create a polydata to store everything in
        polyData = vtkPolyData()

        # Add the points to the dataset
        polyData.SetPoints(points)

        # Add the lines to the dataset
        polyData.SetLines(cells)

        return polyData

    def GetAxis(self, origin, end, ax):
        # Get direction vector for each axis
        direction = [0, 0, 0]
        direction[ax] = 1

        # Construct poly line
        axisLine = self.GetPolyLine(origin, end)

        # Add Tube Filter
        axesTubes = vtkTubeFilter()
        axesTubes.SetInputData(axisLine)
        axesTubes.SetRadius(1 / 10 * self._scale)
        axesTubes.SetNumberOfSides(self._resolution)
        axesTubes.Update()

        # Add Cone as Arrow Tip
        coneSource = vtkConeSource()
        coneSource.SetResolution(self._resolution)
        coneSource.SetRadius(0.3 * self._scale)
        coneSource.SetHeight(1.0 * self._scale)
        coneSource.SetCenter(*end)
        coneSource.SetDirection(*direction)
        coneSource.Update()

        # Combine Tube and Cone to Arrow
        axis = vtkAppendPolyData()
        axis.AddInputData(axesTubes.GetOutput())
        axis.AddInputData(coneSource.GetOutput())
        axis.Update()

        return axis

    def RequestData(self, request, inInfo, outInfo):
        input = dsa.WrapDataObject(vtkImageData.GetData(inInfo[0], 0))
        output1 = dsa.WrapDataObject(vtkPolyData.GetData(outInfo, 0))
        output2 = dsa.WrapDataObject(vtkPolyData.GetData(outInfo, 1))
        output3 = dsa.WrapDataObject(vtkPolyData.GetData(outInfo, 2))

        dimensions = list(input.VTKObject.GetDimensions())
        spacing = list(input.VTKObject.GetSpacing())
        origin = list(input.VTKObject.GetOrigin())

        xend = [origin[0] + dimensions[0] * spacing[0], origin[1], origin[2]]
        yend = [origin[0], origin[1] + dimensions[1] * spacing[1], origin[2]]
        zend = [origin[0], origin[1], origin[2] + dimensions[2] * spacing[2]]

        axis1 = self.GetAxis(origin, xend, 0)
        axis2 = self.GetAxis(origin, yend, 1)
        axis3 = self.GetAxis(origin, zend, 2)

        output1.ShallowCopy(axis1.GetOutput())
        output2.ShallowCopy(axis2.GetOutput())
        output3.ShallowCopy(axis3.GetOutput())

        return 1
