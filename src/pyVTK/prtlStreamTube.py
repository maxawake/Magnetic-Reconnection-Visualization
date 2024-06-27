a__all__ = ['prtlPerpSlice']

from vtkmodules.vtkFiltersParallel import vtkPMaskPoints
from vtkmodules.vtkFiltersCore import vtkTubeFilter
from vtkmodules.vtkFiltersFlowPaths import vtkStreamTracer

from pyprtl.util.vtkAlgorithm import *
from vtkmodules.vtkCommonDataModel import vtkPlane, vtkPolyData, vtkImageData, vtkCellArray, vtkPolyData, vtkPolyLine
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkFiltersCore import vtkCutter
from vtkmodules.numpy_interface import dataset_adapter as dsa
import numpy as np
from vtk import vtkIdList, vtkDoubleArray

@smproxy.filter(label="PRTL Stream Tube")
@smhint_menu('prtl')
@smproperty.input(name='Input2', port_index=1)
@smdomain.datatype(dataTypes=['vtkDataSet'])
@smproperty.input(name='Input1', port_index=0)
@smdomain.datatype(dataTypes=['vtkDataSet']) 
class prtlStreamTube(VTKPythonAlgorithmBase):
    def __init__(self):
        VTKPythonAlgorithmBase.__init__(self, nInputPorts=2, nOutputPorts=1, outputType='vtkPolyData')
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

    def RequestData(self, request, inInfo, outInfo):
        input0 = vtkImageData.GetData(inInfo[0], 0)
        input1 = vtkPolyData.GetData(inInfo[1], 0)
        output = dsa.WrapDataObject(vtkPolyData.GetData(outInfo, 0))

        # create a new vtkTubeFilter
        Tube1 = vtkTubeFilter()
        Tube1.SetCapping(True)
        Tube1.SetDefaultNormal(0.0, 0.0, 1.0)
        Tube1.SetInputDataObject(input1)
        Tube1.SetNumberOfSides(64)
        Tube1.SetRadius(0.3)
        Tube1.SetRadiusFactor(10.0)
        Tube1.SetInputArrayToProcess(0, 0, 0, 0, '')
        Tube1.SetInputArrayToProcess(1, 0, 0, 0, 'Normals')
        Tube1.SetUseDefaultNormal(False)
        Tube1.SetVaryRadius(0)
        Tube1.Update()

        # create a new vtkPMaskPoints
        MaskPoints1 = vtkPMaskPoints()
        MaskPoints1.SetGenerateVertices(False)
        MaskPoints1.SetInputConnection(0, Tube1.GetOutputPort(0))
        MaskPoints1.SetMaximumNumberOfPoints(5000)
        MaskPoints1.SetOffset(0)
        MaskPoints1.SetOnRatio(1)
        MaskPoints1.SetProportionalMaximumNumberOfPoints(False)
        MaskPoints1.SetRandomMode(False)
        MaskPoints1.SetRandomModeType(0)
        MaskPoints1.SetSingleVertexPerCell(False)
        MaskPoints1.Update()

        # create a new vtkStreamTracer
        StreamTracerWithCustomSource1 = vtkStreamTracer()
        StreamTracerWithCustomSource1.SetComputeVorticity(True)
        StreamTracerWithCustomSource1.SetInitialIntegrationStep(0.2)
        StreamTracerWithCustomSource1.AddInputDataObject(input0)
        StreamTracerWithCustomSource1.SetIntegrationDirection(2)
        StreamTracerWithCustomSource1.SetIntegrationStepUnit(2)
        StreamTracerWithCustomSource1.SetIntegratorType(2)
        StreamTracerWithCustomSource1.SetInterpolatorType(0)
        StreamTracerWithCustomSource1.SetMaximumError(1e-06)
        StreamTracerWithCustomSource1.SetMaximumIntegrationStep(0.5)
        StreamTracerWithCustomSource1.SetMaximumNumberOfSteps(2000)
        StreamTracerWithCustomSource1.SetMaximumPropagation(15.75)
        StreamTracerWithCustomSource1.SetMinimumIntegrationStep(0.01)
        StreamTracerWithCustomSource1.SetInputArrayToProcess(0, 0, 0, 0, 'vectors-B')
        StreamTracerWithCustomSource1.SetSourceConnection(MaskPoints1.GetOutputPort(0))
        StreamTracerWithCustomSource1.SetSurfaceStreamlines(False)
        StreamTracerWithCustomSource1.SetTerminalSpeed(1e-12)
        StreamTracerWithCustomSource1.SetUseLocalSeedSource(False)
        StreamTracerWithCustomSource1.Update()

        # create a new vtkTubeFilter
        Tube2 = vtkTubeFilter()
        Tube2.SetCapping(True)
        Tube2.SetDefaultNormal(0.0, 0.0, 1.0)
        Tube2.SetInputConnection(0, StreamTracerWithCustomSource1.GetOutputPort(0))
        Tube2.SetNumberOfSides(6)
        Tube2.SetRadius(0.09451468769973144)
        Tube2.SetRadiusFactor(10.0)
        Tube2.SetInputArrayToProcess(0, 0, 0, 0, 'scalars-rhoI')
        Tube2.SetInputArrayToProcess(1, 0, 0, 0, 'Normals')
        Tube2.SetUseDefaultNormal(False)
        Tube2.SetVaryRadius(0)
        Tube2.Update()

        output.ShallowCopy(Tube2.GetOutput())

        return 1