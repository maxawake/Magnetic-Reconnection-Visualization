__all__ = ['prtlPseudoBifurcationLine']

from pyprtl.util.vtkAlgorithm import *
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkDataSet, vtkDataObject, vtkPolyData
from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtkmodules.numpy_interface import dataset_adapter as dsa
from vtkmodules.vtkCommonCore import vtkPoints


@smproxy.filter(label="PRTL Pseudo Bifurcation Lines")
@smhint_menu('prtl')
@smproperty.input(name='Input', port_index=0)
@smdomain.datatype(dataTypes=['vtkDataSet']) 
class prtlPseudoBifurcationLine(VTKPythonAlgorithmBase):
    def __init__(self):
        self._array_field = 0
        self._array_name = None
        VTKPythonAlgorithmBase.__init__(self, nInputPorts=1, nOutputPorts=1, outputType='vtkPolyData')

    @smproperty_inputarray('Vectors', attribute_type='Vectors')
    def SetInputArrayToProcess(self, idx, port, connection, field, name):
        self._array_field = field
        self._array_name = name
        self.Modified()

    def RequestDataObject(self, request, inInfo, outInfo):
        inp = vtkDataSet.GetData(inInfo[0], 0)
        if not inp:
            return 0
        for i in range(self.GetNumberOfOutputPorts()):
            output = vtkDataSet.GetData(outInfo, i)
            if not output or not output.IsA(inp.GetClassName()):
                outInfo.GetInformationObject(i).Set(
                    vtkDataObject.DATA_OBJECT(), inp.NewInstance())
        return 1

    def RequestInformation(self, request, inInfo, outInfo):
        executive = self.GetExecutive()
        in_info = inInfo[0].GetInformationObject(0)
        if not in_info.Has(executive.WHOLE_EXTENT()):
            return 1

        extent = list(in_info.Get(executive.WHOLE_EXTENT()))
        dims = [extent[2 * i + 1] - extent[2 * i] + 1 for i in range(len(extent) // 2)]

        out_info = outInfo.GetInformationObject(0)
        out_info.Set(executive.WHOLE_EXTENT(), extent, 6)
        return 1

    def RequestUpdateExtent(self, request, inInfo, outInfo):
        executive = self.GetExecutive()
        in_info = inInfo[0].GetInformationObject(0)
        in_info.Set(executive.UPDATE_EXTENT(), in_info.Get(executive.WHOLE_EXTENT()), 6)
        return 1

    def RequestData(self, request, inInfo, outInfo):
        input = vtkPolyData.GetData(inInfo[0], 0)
        output = dsa.WrapDataObject(vtkPolyData.GetData(outInfo, 0))

        # Get the scalar array
        scalar_array = input.GetPointData().GetArray('FeatureStrength')

        # Create a new poly data object
        new_poly_data = vtkPolyData()

        # Create a new points array to store the highest scalar value points
        points = vtkPoints()

        # Iterate over the lines
        for line_id in range(input.GetNumberOfLines()):
            line = input.GetCell(line_id)
            line_points = line.GetPointIds()

            # Find the point with the highest scalar value on the line
            max_scalar_value = -float('inf')
            max_scalar_point_id = -1
            for i in range(line_points.GetNumberOfIds()):
                point_id = line_points.GetId(i)
                scalar_value = scalar_array.GetValue(point_id)
                if scalar_value > max_scalar_value:
                    max_scalar_value = scalar_value
                    max_scalar_point_id = point_id

            # Add the point with the highest scalar value to the new points array
            point = input.GetPoint(max_scalar_point_id)
            points.InsertNextPoint(point)

        # Set the new points array to the new poly data object
        new_poly_data.SetPoints(points)

        # Set the output to the new poly data object
        output.ShallowCopy(new_poly_data)

        print("Done.")
        return 1