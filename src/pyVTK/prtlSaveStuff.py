__all__ = ['prtlSaveStuff']

from pyprtl.util.vtkAlgorithm import *
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkDataSet, vtkDataObject, vtkPolyData
from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtkmodules.numpy_interface import dataset_adapter as dsa
from vtkmodules.vtkCommonCore import vtkPoints
import os
import paraview.simple as pv

@smproxy.filter(label="PRTL Save Stuff")
@smhint_menu('prtl')
@smproperty.input(name='Input', port_index=0)
@smdomain.datatype(dataTypes=['vtkDataSet']) 
class prtlSaveStuff(VTKPythonAlgorithmBase):
    def __init__(self):
        self._array_field = 0
        self._array_name = None
        self._save_path = None  # Added save path variable
        VTKPythonAlgorithmBase.__init__(self, nInputPorts=1, nOutputPorts=1, outputType='vtkPolyData')

    @smproperty_inputarray('Vectors', attribute_type='Vectors')
    def SetInputArrayToProcess(self, idx, port, connection, field, name):
        self._array_field = field
        self._array_name = name
        self.Modified()
    
    @smproperty.stringvector(name='Save Path')  # Added save path property
    def SetSavePath(self, path):
        self._save_path = path

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

        # Get the current time step
        time_step = pv.GetAnimationScene().TimeKeeper.Time

        # Save the number of points to a text file
        num_points = input.GetNumberOfPoints()
        save_path = os.path.join(self._save_path, f'num_points_{time_step}.txt')  # Use save path variable
        with open(save_path, 'w') as file:
            file.write(str(num_points))

        # Save the number of lines to a text file
        num_lines = input.GetNumberOfLines()
        save_path = os.path.join(self._save_path, f'num_lines_{time_step}.txt')  # Use save path variable
        with open(save_path, 'w') as file:
            file.write(str(num_lines))

        # Save the point values of the "Result" array to a text file
        result_array = input.GetPointData().GetArray('Result')
        if result_array:
            num_values = result_array.GetNumberOfTuples()
            save_path = os.path.join(self._save_path, f'result_values_{time_step}.txt')  # Use save path variable
            with open(save_path, 'w') as file:
                for i in range(num_values):
                    value = result_array.GetTuple(i)
                    file.write(','.join(str(v) for v in value) + '\n')

        print("Done.")
        return 1
