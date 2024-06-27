a__all__ = ['prtlPerpSlice']
from pyprtl.util.vtkAlgorithm import *
from vtkmodules.vtkCommonDataModel import vtkPlane, vtkPolyData, vtkImageData, vtkCellArray, vtkPolyData, vtkPolyLine
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkFiltersCore import vtkCutter
from vtkmodules.numpy_interface import dataset_adapter as dsa
import numpy as np
from vtk import vtkIdList, vtkDoubleArray

@smproxy.filter(label="PRTL Perpendicular Slice")
@smhint_menu('prtl')
@smproperty.input(name='Input2', port_index=1)
@smdomain.datatype(dataTypes=['vtkDataSet'])
@smproperty.input(name='Input1', port_index=0)
@smdomain.datatype(dataTypes=['vtkDataSet']) 
class prtlPerpSlice(VTKPythonAlgorithmBase):
    def __init__(self):
        VTKPythonAlgorithmBase.__init__(self, nInputPorts=2, nOutputPorts=2, outputType='vtkPolyData')
        self._array_field = [0] * 2
        self._array_name = [None] * 2
        self._line_segment = 0.0
        self._cell_id = 0
    
    @smproperty_inputarray('Data', idx=0, input_name='Input1', attribute_type='Vectors')
    def SetInputArrayToProcessA(self, idx, port, connection, field, name):
        self._array_field[0] = field
        self._array_name[0] = name
        self.Modified()

    @smproperty_inputarray('Line', idx=1, input_name='Input2', attribute_type='PolyData')
    def SetInputArrayToProcessB(self, idx, port, connection, field, name):
        self._array_field[1] = field
        self._array_name[1] = name
        self.Modified()

    @smproperty.doublevector(name='LineSegment', label='Line Segment', default_values=0.0)
    @smdomain.doublerange(min=0.0, max=1.0)
    def SetLineSegment(self, d):
        self._line_segment = d
        self.Modified()

    @smproperty.intvector(name='CellId', label='Cell Identification', default_values=0)
    @smdomain.intrange(min=0, max=100)
    def SetCellId(self, i):
        self._cell_id = i
        self.Modified()

    def RequestData(self, request, inInfo, outInfo):
        input0 = dsa.WrapDataObject(vtkImageData.GetData(inInfo[0], 0))
        input1 = dsa.WrapDataObject(vtkPolyData.GetData(inInfo[1], 0))
        output0 = dsa.WrapDataObject(vtkPolyData.GetData(outInfo, 0))
        output1 = dsa.WrapDataObject(vtkPolyData.GetData(outInfo, 1))
        
        # # # Version 1 # # #
        # inp_points = input1.GetPoints()
        # npoints = input1.GetNumberOfPoints()-1
        # current_idx = int(npoints*self._line_segment)
        # if current_idx == npoints:
        #     current_idx -= 1

        # normal = inp_points[current_idx+1] - inp_points[current_idx]
        # current_point = inp_points[current_idx]


        # Get Cell Id List and poulate with Ids from PolyData
        cellIds = vtkIdList() 
        currentCellId = np.clip(self._cell_id, 0, input1.GetNumberOfCells()-1)
        input1.GetCellPoints(currentCellId, cellIds)

        # Get current index
        npoints = cellIds.GetNumberOfIds() - 1
        current_idx = int(npoints*self._line_segment)
        if current_idx == npoints:
            current_idx -= 1

        # Get consecutive points in cell to calculate normal
        current_point=np.array(input1.GetPoint(cellIds.GetId(current_idx)))
        next_point=np.array(input1.GetPoint(cellIds.GetId(current_idx+1)))

        # Calculate normal
        normal = next_point - current_point

        # Define plane with normal to cut dataset with
        plane = vtkPlane()
        plane.SetOrigin(current_point[0], current_point[1], current_point[2])
        plane.SetNormal(normal[0], normal[1], normal[2])

        # Define cutter with plane 
        planeCut = vtkCutter()
        planeCut.SetInputData(input0.VTKObject)
        planeCut.SetCutFunction(plane)
        planeCut.Update()

        output0.ShallowCopy(planeCut.GetOutput())

        # Create a vtkPoints object and store the points in it
        points = vtkPoints()
        points.InsertNextPoint(current_point)
        points.InsertNextPoint(next_point)

        polyLine = vtkPolyLine()
        polyLine.GetPointIds().SetNumberOfIds(2)
        for i in range(0,2):
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

        # Create array to store normal vector in for visualization
        array = vtkDoubleArray()
        array.SetNumberOfComponents(3)
        array.SetName("Normals")
        array.SetNumberOfComponents(3)
        array.SetNumberOfTuples(2)
        array.SetTuple3(0, normal[0], normal[1], normal[2])
        array.SetTuple3(1, normal[0], normal[1], normal[2])

        # Attach normal vector to current point
        polyData.GetPointData().SetVectors(array)

        output1.ShallowCopy(polyData)
        return 1





