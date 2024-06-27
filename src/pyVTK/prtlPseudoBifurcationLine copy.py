__all__ = ['prtlPseudoBifurcationLine']

from pyprtl.maxpy.prtlDirectionalDerivative import prtlDirectionalDerivative
from pyprtl.prtlGaussianFilter2 import prtlGaussianFilter2
from pyprtl.dependent_vectors.prtlParallelVectorsOperator import prtlParallelVectorsOperator
from paraview.modules.vtkPVVTKExtensionsMisc import vtkPVPlane
from prtl.vtk.prtlPolyDataUnstructuredGridConversion import prtlUnstructuredGridToPolyData
from prtl.vtk.prtlVectorFieldDerivatives import prtlVectorFieldDerivatives
from paraview.modules.vtkPVVTKExtensionsFiltersGeneral import vtkPVArrayCalculator
from vtkmodules.vtkFiltersCore import vtkResampleWithDataSet
from paraview.modules.vtkPVVTKExtensionsIOCore import vtkFileSeriesReader
from vtkmodules.vtkCommonDataModel import vtkStaticCellLocator
from vtkmodules.vtkIOXML import vtkXMLImageDataReader
from vtkmodules.vtkFiltersFlowPaths import vtkStreamTracer
from paraview.modules.vtkPVVTKExtensionsFiltersGeneral import vtkPVMetaClipDataSet

from pyprtl.util.vtkAlgorithm import *
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkDataSet, vtkDataObject, vtkPolyData
from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtkmodules.numpy_interface import dataset_adapter as dsa
import math
import numpy as np

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
        input = vtkImageData.GetData(inInfo[0], 0)
        output = dsa.WrapDataObject(vtkPolyData.GetData(outInfo, 0))

        output.ShallowCopy(input)

        print("Filtering...")
        # create a new prtlGaussianFilter2
        PRTLGaussianFilter21 = prtlGaussianFilter2()
        PRTLGaussianFilter21.SetInputDataObject(input)
        PRTLGaussianFilter21.AddPointDataArray('vectors-B')
        PRTLGaussianFilter21.AddPointDataArray('vectors-E')
        PRTLGaussianFilter21.AddPointDataArray('vectors-rhoI')
        PRTLGaussianFilter21.SetSigma(2.0)
        PRTLGaussianFilter21.SetSigmaIsCellLength(False)
        PRTLGaussianFilter21.SetSuffix('')
        PRTLGaussianFilter21.Update()

        print("Derivative...")
        # create a new prtlDirectionalDerivative
        PRTLDirectionalDerivative1 = prtlDirectionalDerivative()
        PRTLDirectionalDerivative1.SetInputConnection(0, PRTLGaussianFilter21.GetOutputPort(0))
        PRTLDirectionalDerivative1.SetInputArrayToProcess(0, 0, 0, 0, 'vectors-B')
        PRTLDirectionalDerivative1.Update()

        print("Parallel Vectors...")
        # create a new prtlParallelVectorsOperator
        PRTLParallelVectorsOperator1 = prtlParallelVectorsOperator()
        PRTLParallelVectorsOperator1.SetAngleNeighborhood(1)
        PRTLParallelVectorsOperator1.SetFeatureStrength(3)
        PRTLParallelVectorsOperator1.SetFilterByAngle(True)
        PRTLParallelVectorsOperator1.SetFilterByScalars(False)
        PRTLParallelVectorsOperator1.SetFilterByVertexCount(True)
        PRTLParallelVectorsOperator1.SetFilterByVolume(False)
        PRTLParallelVectorsOperator1.SetIncludeBoundary(False)
        PRTLParallelVectorsOperator1.SetInputConnection(0, PRTLDirectionalDerivative1.GetOutputPort(0))
        PRTLParallelVectorsOperator1.SetMaxAngle(15.0)
        PRTLParallelVectorsOperator1.SetMinFeatureStrength(0.0)
        PRTLParallelVectorsOperator1.SetMinVertexCount(20)
        PRTLParallelVectorsOperator1.SetMinVolume(0.0)
        PRTLParallelVectorsOperator1.SetNewtonSteps(20)
        PRTLParallelVectorsOperator1.SetPseudoVectorsU(False)
        PRTLParallelVectorsOperator1.SetPseudoVectorsW(False)
        PRTLParallelVectorsOperator1.SetScalarRange(-1e+300, 1e+300)
        PRTLParallelVectorsOperator1.SetInputArrayToProcess(0, 0, 0, 0, 'vectors-B')
        PRTLParallelVectorsOperator1.SetInputArrayToProcess(1, 0, 0, 0, 'Directional Derivative')
        PRTLParallelVectorsOperator1.SetInputArrayToProcess(4, 0, 0, 0, 'None')
        PRTLParallelVectorsOperator1.SetInputArrayToProcess(5, 0, 0, 0, 'None')
        PRTLParallelVectorsOperator1.Update()

        # create a new prtlUnstructuredGridToPolyData
        PRTLUnstructuredGridToPolyData1 = prtlUnstructuredGridToPolyData()
        PRTLUnstructuredGridToPolyData1.SetCopyCellData(True)
        PRTLUnstructuredGridToPolyData1.SetCopyFieldData(True)
        PRTLUnstructuredGridToPolyData1.SetCopyPointData(True)
        PRTLUnstructuredGridToPolyData1.SetIgnoreUnsupportedCells(False)
        PRTLUnstructuredGridToPolyData1.SetInputConnection(0, PRTLParallelVectorsOperator1.GetOutputPort(0))
        PRTLUnstructuredGridToPolyData1.Update()

        print("Integrate Streamlines...")
        # create a new vtkStreamTracer
        StreamTracerWithCustomSource1 = vtkStreamTracer()
        StreamTracerWithCustomSource1.SetComputeVorticity(True)
        StreamTracerWithCustomSource1.SetInitialIntegrationStep(0.2)
        StreamTracerWithCustomSource1.AddInputDataObject(input)#Connection(0, input.GetOutputPort(0))
        StreamTracerWithCustomSource1.SetIntegrationDirection(2)
        StreamTracerWithCustomSource1.SetIntegrationStepUnit(2)
        StreamTracerWithCustomSource1.SetIntegratorType(2)
        StreamTracerWithCustomSource1.SetInterpolatorType(0)
        StreamTracerWithCustomSource1.SetMaximumError(1e-06)
        StreamTracerWithCustomSource1.SetMaximumIntegrationStep(0.5)
        StreamTracerWithCustomSource1.SetMaximumNumberOfSteps(2000)
        StreamTracerWithCustomSource1.SetMaximumPropagation(255.0)
        StreamTracerWithCustomSource1.SetMinimumIntegrationStep(0.01)
        StreamTracerWithCustomSource1.SetInputArrayToProcess(0, 0, 0, 0, 'vectors-B')
        StreamTracerWithCustomSource1.SetSourceConnection(PRTLUnstructuredGridToPolyData1.GetOutputPort(0))
        StreamTracerWithCustomSource1.SetSurfaceStreamlines(False)
        StreamTracerWithCustomSource1.SetTerminalSpeed(1e-12)
        StreamTracerWithCustomSource1.SetUseLocalSeedSource(False)
        StreamTracerWithCustomSource1.Update()

        print("Eigendecomposition...")
        # create a new prtlVectorFieldDerivatives
        PRTLVectorFieldDerivatives1 = prtlVectorFieldDerivatives()
        PRTLVectorFieldDerivatives1.SetComputeAcceleration(False)
        PRTLVectorFieldDerivatives1.SetComputeEigenDecomposition(True)
        PRTLVectorFieldDerivatives1.SetComputeFeatureFlowField(False)
        PRTLVectorFieldDerivatives1.SetComputeStrainEigenDecomposition(False)
        PRTLVectorFieldDerivatives1.SetInputDataObject(input)#Connection(0, input.GetOutputPort(0))
        PRTLVectorFieldDerivatives1.SetLeastSquaresDerivatives(False)
        PRTLVectorFieldDerivatives1.SetLeastSquaresRadius(2)
        PRTLVectorFieldDerivatives1.SetOutputDoublePrecision(True)
        PRTLVectorFieldDerivatives1.SetOutputStrainTensor(False)
        PRTLVectorFieldDerivatives1.SetOutputStructuredGrid(False)
        PRTLVectorFieldDerivatives1.SetInputArrayToProcess(0, 0, 0, 0, 'vectors-B')
        PRTLVectorFieldDerivatives1.Update()

        print("Feature Strength...")
        # create a new vtkPVArrayCalculator
        Calculator1 = vtkPVArrayCalculator()
        Calculator1.SetAttributeType(0)
        Calculator1.SetCoordinateResults(False)
        Calculator1.SetFunction('-(RealEigenvalueMajor*RealEigenvalueMinor)')
        Calculator1.SetFunctionParserTypeFromInt(1)
        Calculator1.SetInputConnection(0, PRTLVectorFieldDerivatives1.GetOutputPort(0))
        Calculator1.SetReplaceInvalidValues(True)
        Calculator1.SetReplacementValue(0.0)
        Calculator1.SetResultArrayName('Result')
        Calculator1.SetResultArrayType(11)
        Calculator1.SetResultNormals(False)
        Calculator1.SetResultTCoords(False)
        Calculator1.Update()

        print("Resample...")
        # create a new vtkResampleWithDataSet
        ResampleWithDataset1 = vtkResampleWithDataSet()
        ResampleWithDataset1.SetCategoricalData(False)

        # create a new vtkStaticCellLocator
        ResampleWithDataset1_CellLocator = vtkStaticCellLocator()
        ResampleWithDataset1.SetCellLocatorPrototype(ResampleWithDataset1_CellLocator)
        ResampleWithDataset1.SetComputeTolerance(True)
        ResampleWithDataset1.SetInputConnection(0, StreamTracerWithCustomSource1.GetOutputPort(0))
        ResampleWithDataset1.SetMarkBlankPointsAndCells(False)
        ResampleWithDataset1.SetPassCellArrays(False)
        ResampleWithDataset1.SetPassFieldArrays(True)
        ResampleWithDataset1.SetPassPartialArrays(False)
        ResampleWithDataset1.SetPassPointArrays(False)
        ResampleWithDataset1.SetSnapToCellWithClosestPoint(False)
        ResampleWithDataset1.SetSourceConnection(Calculator1.GetOutputPort(0))
        ResampleWithDataset1.SetTolerance(2.220446049250313e-16)
        ResampleWithDataset1.Update()

        print("Filter....")
        # create a new vtkPVMetaClipDataSet
        Clip1 = vtkPVMetaClipDataSet()
        Clip1.SetDataSetClipFunction(None)
        Clip1.SetExactBoxClip(False)

        # create a new vtkPVPlane
        Clip1_HyperTreeGridClipFunction = vtkPVPlane()
        Clip1_HyperTreeGridClipFunction.SetNormal(1.0, 0.0, 0.0)
        Clip1_HyperTreeGridClipFunction.SetOffset(0.0)
        Clip1_HyperTreeGridClipFunction.SetOrigin(127.49999490007758, 127.49993971735239, 127.50000850111246)
        Clip1.SetHyperTreeGridClipFunction(Clip1_HyperTreeGridClipFunction)
        Clip1.SetInputConnection(0, ResampleWithDataset1.GetOutputPort(0))
        Clip1.SetInsideOut(False)
        Clip1.PreserveInputCells(False)
        Clip1.SetInputArrayToProcess(0, 0, 0, 0, 'Result')
        Clip1.SetUseValueAsOffset(False)
        Clip1.SetValue(0.000835152916582154)
        Clip1.Update()

        # create a new prtlUnstructuredGridToPolyData
        PRTLUnstructuredGridToPolyData2 = prtlUnstructuredGridToPolyData()
        PRTLUnstructuredGridToPolyData2.SetCopyCellData(True)
        PRTLUnstructuredGridToPolyData2.SetCopyFieldData(True)
        PRTLUnstructuredGridToPolyData2.SetCopyPointData(True)
        PRTLUnstructuredGridToPolyData2.SetIgnoreUnsupportedCells(False)
        PRTLUnstructuredGridToPolyData2.SetInputConnection(0, Clip1.GetOutputPort(0))
        PRTLUnstructuredGridToPolyData2.Update()

        output.ShallowCopy(PRTLUnstructuredGridToPolyData2.GetOutput())
        print("Done.")
        return 1