a__all__ = ['prtlBifurcationLine']
from pyprtl.util.vtkAlgorithm import *
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkPolyData
from vtkmodules.numpy_interface import dataset_adapter as dsa
from pyprtl.dependent_vectors.prtlParallelVectorsOperator import prtlParallelVectorsOperator
from paraview.modules.vtkPVVTKExtensionsFiltersGeneral import vtkPVArrayCalculator
from vtkmodules.vtkFiltersGeneral import vtkPassSelectedArrays
from prtl.vtk.prtlPolyDataUnstructuredGridConversion import prtlUnstructuredGridToPolyData
from prtl.vtk.prtlEigenvectors import prtlEigenvectors
from vtkmodules.vtkFiltersCore import vtkStripper
from prtl.vtk.prtlOrientEigenvectors import prtlOrientEigenvectors
from prtl.vtk.prtlPolyDataUnstructuredGridConversion import prtlPolyDataToUnstructuredGrid
from vtkmodules.vtkFiltersGeneral import vtkWarpVector


@smproxy.filter(label="PRTL Bifurcation Manifold")
@smhint_menu('prtl')
@smproperty.input(name='Input', port_index=0)
@smdomain.datatype(dataTypes=['vtkDataSet'])
class prtlBifurcationLine(VTKPythonAlgorithmBase):
    def __init__(self):
        self._array_field = 0
        self._array_name = None
        self._offset = 0.1
        VTKPythonAlgorithmBase.__init__(self,
                                        nInputPorts=1, nOutputPorts=4,
                                        inputType='vtkImageData', outputType='vtkPolyData')

    @smproperty_inputarray('Vectors', attribute_type='Vectors')
    def SetInputArrayToProcess(self, idx, port, connection, field, name):
        self._array_field = field
        self._array_name = name
        self.Modified()

    @smproperty.doublevector(name='Offset', label='Offset', default_values=0.1)
    @smdomain.doublerange(min=0.0)
    def SetOffset(self, d):
        self._offset = d
        self.Modified()
    
    def GetOffsetLine(self, Eigenvectors, index, offset):
        if index == 0:
            FunctionString = 'iHat*"EigenvectorReal_0"+jHat*"EigenvectorReal_1"+kHat*"EigenvectorReal_2"'
        if index == 1:
            FunctionString = 'iHat*"EigenvectorReal_3"+jHat*"EigenvectorReal_4"+kHat*"EigenvectorReal_5"'
        if index == 2:
            FunctionString = 'iHat*"EigenvectorReal_6"+jHat*"EigenvectorReal_7"+kHat*"EigenvectorReal_8"'

        # create a new vtkPVArrayCalculator
        Calculator1 = vtkPVArrayCalculator()
        Calculator1.SetAttributeType(0)
        Calculator1.SetCoordinateResults(False)
        Calculator1.SetFunction(FunctionString)
        Calculator1.SetFunctionParserTypeFromInt(1)
        Calculator1.SetInputConnection(0, Eigenvectors.GetOutputPort(0))
        Calculator1.SetReplaceInvalidValues(True)
        Calculator1.SetReplacementValue(0.0)
        Calculator1.SetResultArrayName('Result')
        Calculator1.SetResultArrayType(11)
        Calculator1.SetResultNormals(False)
        Calculator1.SetResultTCoords(False)
        Calculator1.Update()

        # create a new vtkWarpVector
        WarpByVector1 = vtkWarpVector()
        WarpByVector1.SetInputConnection(0, Calculator1.GetOutputPort(0))
        WarpByVector1.SetScaleFactor(offset)
        WarpByVector1.SetInputArrayToProcess(0, 0, 0, 0, 'Result')
        WarpByVector1.Update()

        # create a new vtkPassSelectedArrays
        PassArrays1 = vtkPassSelectedArrays()
        PassArrays1.GetFieldDataArraySelection().EnableArray('Count')
        PassArrays1.GetFieldDataArraySelection().EnableArray('Dimensions')
        PassArrays1.GetFieldDataArraySelection().EnableArray('NumberOfDependentVectors')
        PassArrays1.GetFieldDataArraySelection().EnableArray('NumberOfFaceClasses')
        PassArrays1.GetFieldDataArraySelection().EnableArray('Origin')
        PassArrays1.GetFieldDataArraySelection().EnableArray('Spacing')
        PassArrays1.GetFieldDataArraySelection().EnableArray('Volume')
        PassArrays1.SetInputConnection(0, WarpByVector1.GetOutputPort(0))
        PassArrays1.Update()

        # create a new prtlUnstructuredGridToPolyData
        PRTLUnstructuredGridToPolyData1 = prtlUnstructuredGridToPolyData()
        PRTLUnstructuredGridToPolyData1.SetCopyCellData(True)
        PRTLUnstructuredGridToPolyData1.SetCopyFieldData(True)
        PRTLUnstructuredGridToPolyData1.SetCopyPointData(True)
        PRTLUnstructuredGridToPolyData1.SetIgnoreUnsupportedCells(False)
        PRTLUnstructuredGridToPolyData1.SetInputConnection(0, PassArrays1.GetOutputPort(0))
        PRTLUnstructuredGridToPolyData1.Update()

        return PRTLUnstructuredGridToPolyData1

    def RequestData(self, request, inInfo, outInfo):
        input = dsa.WrapDataObject(vtkImageData.GetData(inInfo[0], 0))
        output0 = dsa.WrapDataObject(vtkPolyData.GetData(outInfo, 0))
        output1 = dsa.WrapDataObject(vtkPolyData.GetData(outInfo, 1))
        output2 = dsa.WrapDataObject(vtkPolyData.GetData(outInfo, 2))
        output3 = dsa.WrapDataObject(vtkPolyData.GetData(outInfo, 3))

        # create a new prtlParallelVectorsOperator
        PRTLParallelVectorsOperator1 = prtlParallelVectorsOperator()
        PRTLParallelVectorsOperator1.SetAngleNeighborhood(1)
        PRTLParallelVectorsOperator1.SetFeatureStrength(3)
        PRTLParallelVectorsOperator1.SetFilterByAngle(True)
        PRTLParallelVectorsOperator1.SetFilterByScalars(False)
        PRTLParallelVectorsOperator1.SetFilterByVertexCount(False)
        PRTLParallelVectorsOperator1.SetFilterByVolume(False)
        PRTLParallelVectorsOperator1.SetIncludeBoundary(False)
        PRTLParallelVectorsOperator1.SetInputDataObject(input.VTKObject)
        PRTLParallelVectorsOperator1.SetMaxAngle(15.0)
        PRTLParallelVectorsOperator1.SetMinFeatureStrength(0.0)
        PRTLParallelVectorsOperator1.SetMinVertexCount(1)
        PRTLParallelVectorsOperator1.SetMinVolume(0.0)
        PRTLParallelVectorsOperator1.SetNewtonSteps(20)
        PRTLParallelVectorsOperator1.SetPseudoVectorsU(False)
        PRTLParallelVectorsOperator1.SetPseudoVectorsW(False)
        PRTLParallelVectorsOperator1.SetScalarRange(-1e+300, 1e+300)
        PRTLParallelVectorsOperator1.SetInputArrayToProcess(0, 0, 0, 0, 'vectors')
        PRTLParallelVectorsOperator1.SetInputArrayToProcess(1, 0, 0, 0, 'Jac')
        PRTLParallelVectorsOperator1.SetInputArrayToProcess(4, 0, 0, 0, 'None')
        PRTLParallelVectorsOperator1.SetInputArrayToProcess(5, 0, 0, 0, 'None')
        PRTLParallelVectorsOperator1.Update()

        if PRTLParallelVectorsOperator1.GetOutputDataObject(0).GetNumberOfPoints() == 0:
            return 1

        # create a new prtlUnstructuredGridToPolyData
        PRTLUnstructuredGridToPolyData1 = prtlUnstructuredGridToPolyData()
        PRTLUnstructuredGridToPolyData1.SetCopyCellData(True)
        PRTLUnstructuredGridToPolyData1.SetCopyFieldData(True)
        PRTLUnstructuredGridToPolyData1.SetCopyPointData(True)
        PRTLUnstructuredGridToPolyData1.SetIgnoreUnsupportedCells(False)
        PRTLUnstructuredGridToPolyData1.SetInputConnection(0, PRTLParallelVectorsOperator1.GetOutputPort(0))
        PRTLUnstructuredGridToPolyData1.Update()


        # create a new vtkStripper
        TriangleStrips1 = vtkStripper()
        TriangleStrips1.SetInputConnection(0, PRTLUnstructuredGridToPolyData1.GetOutputPort(0))
        TriangleStrips1.SetJoinContiguousSegments(True)
        TriangleStrips1.SetMaximumLength(100000)
        TriangleStrips1.Update()

        # create a new vtkStripper
        TriangleStrips2 = vtkStripper()
        TriangleStrips2.SetInputConnection(0, TriangleStrips1.GetOutputPort(0))
        TriangleStrips2.SetJoinContiguousSegments(True)
        TriangleStrips2.SetMaximumLength(1000)
        TriangleStrips2.Update()

        # create a new vtkStripper
        TriangleStrips3 = vtkStripper()
        TriangleStrips3.SetInputConnection(0, TriangleStrips2.GetOutputPort(0))
        TriangleStrips3.SetJoinContiguousSegments(True)
        TriangleStrips3.SetMaximumLength(1000)
        TriangleStrips3.Update()

        # create a new vtkStripper
        TriangleStrips4 = vtkStripper()
        TriangleStrips4.SetInputConnection(0, TriangleStrips3.GetOutputPort(0))
        TriangleStrips4.SetJoinContiguousSegments(True)
        TriangleStrips4.SetMaximumLength(1000)
        TriangleStrips4.Update()

        # create a new prtlPolyDataToUnstructuredGrid
        PRTLPolyDataToUnstructuredGrid1 = prtlPolyDataToUnstructuredGrid()
        PRTLPolyDataToUnstructuredGrid1.SetCopyCellData(True)
        PRTLPolyDataToUnstructuredGrid1.SetCopyFieldData(True)
        PRTLPolyDataToUnstructuredGrid1.SetCopyPointData(True)
        PRTLPolyDataToUnstructuredGrid1.SetInputConnection(0, TriangleStrips4.GetOutputPort(0))

        # create a new prtlEigenvectors
        PRTLEigenvectors1 = prtlEigenvectors()
        PRTLEigenvectors1.SetInputDataObject(input.VTKObject)
        PRTLEigenvectors1.SetInputConnection(1, PRTLPolyDataToUnstructuredGrid1.GetOutputPort(0))
        PRTLEigenvectors1.SetInputArrayToProcess(0, 0, 0, 0, 'vectors')
        PRTLEigenvectors1.Update()

        # create a new prtlOrientEigenvectors
        PRTLOrientEigenvectors1 = prtlOrientEigenvectors()
        PRTLOrientEigenvectors1.SetDimension(-1)
        PRTLOrientEigenvectors1.SetInputConnection(0, PRTLEigenvectors1.GetOutputPort(0))
        PRTLOrientEigenvectors1.SetInputArrayToProcess(0, 0, 0, 0, 'EigenvalueReal')
        PRTLOrientEigenvectors1.SetInputArrayToProcess(1, 0, 0, 0, 'EigenvectorReal')
        PRTLOrientEigenvectors1.SetSortByEigenvalue(True)
        PRTLOrientEigenvectors1.Update()

        # Create four offset lines
        PRTLUnstructuredGridToPolyData0 = self.GetOffsetLine(PRTLOrientEigenvectors1, 0, self._offset)
        PRTLUnstructuredGridToPolyData1 = self.GetOffsetLine(PRTLOrientEigenvectors1, 0, -self._offset)
        PRTLUnstructuredGridToPolyData2 = self.GetOffsetLine(PRTLOrientEigenvectors1, 2, self._offset)
        PRTLUnstructuredGridToPolyData3 = self.GetOffsetLine(PRTLOrientEigenvectors1, 2, -self._offset)

        # Copy to output 
        output0.ShallowCopy(PRTLUnstructuredGridToPolyData0.GetOutput())
        output1.ShallowCopy(PRTLUnstructuredGridToPolyData1.GetOutput())
        output2.ShallowCopy(PRTLUnstructuredGridToPolyData2.GetOutput())
        output3.ShallowCopy(PRTLUnstructuredGridToPolyData3.GetOutput())

        return 1





