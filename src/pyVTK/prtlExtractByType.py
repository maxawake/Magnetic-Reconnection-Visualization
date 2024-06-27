__all__ = ["prtlExtractByType"]

from pyprtl.util.vtkAlgorithm import *
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkDataSet, vtkDataObject, vtkPolyData
from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtkmodules.numpy_interface import dataset_adapter as dsa
from vtkmodules.vtkCommonCore import vtkPoints
import os
import paraview.simple as pv


@smproxy.filter(label="PRTL Extract By Type")
@smhint_menu("prtl")
@smproperty.input(name="Input", port_index=0)
@smdomain.datatype(dataTypes=["vtkDataSet"])
class prtlExtractByType(VTKPythonAlgorithmBase):
    def __init__(self):
        self._array_field = 0
        self._array_name = None
        self._save_path = None  # Added save path variable
        VTKPythonAlgorithmBase.__init__(self, nInputPorts=1, nOutputPorts=1, outputType="vtkPolyData")

    @smproperty_inputarray("Vectors", attribute_type="Vectors")
    def SetInputArrayToProcess(self, idx, port, connection, field, name):
        self._array_field = field
        self._array_name = name
        self.Modified()

    @smproperty.intvector(name="Type ID", label="Type ID", default_values=0)
    @smdomain.intrange(min=0, max=6)
    def SetTypeID(self, i):
        self._typeID = i
        self.Modified()

    def RequestDataObject(self, request, inInfo, outInfo):
        inp = vtkDataSet.GetData(inInfo[0], 0)
        if not inp:
            return 0
        for i in range(self.GetNumberOfOutputPorts()):
            output = vtkDataSet.GetData(outInfo, i)
            if not output or not output.IsA(inp.GetClassName()):
                outInfo.GetInformationObject(i).Set(vtkDataObject.DATA_OBJECT(), inp.NewInstance())
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

        # Get the point data
        point_data = input.GetPointData()

        # Get the array containing the point ids
        point_ids_array = point_data.GetArray("typeDetailed")

        # Create a new polydata to store the extracted points
        extracted_polydata = vtkPolyData()

        # Create a new points array to store the extracted points
        extracted_points = vtkPoints()

        # Iterate over all points and extract the ones with the desired id
        for i in range(input.GetNumberOfPoints()):
            point_id = point_ids_array.GetValue(i)
            if point_id == self._typeID:
                point = input.GetPoint(i)
                extracted_points.InsertNextPoint(point)

        # Set the extracted points as the points of the new polydata
        extracted_polydata.SetPoints(extracted_points)

        # Set the output polydata
        output.ShallowCopy(extracted_polydata)
        print("Done.")
        return 1
