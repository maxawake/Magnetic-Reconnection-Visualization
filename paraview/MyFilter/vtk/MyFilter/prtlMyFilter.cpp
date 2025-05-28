#include "prtlMyFilter.h"

#include <prtl/vtk/detail/default_request_data_object.h>

#include <vtkCommand.h>
#include <vtkInformation.h>
#include <vtkInformationVector.h>
#include <vtkObjectFactory.h>
#include <vtkStreamingDemandDrivenPipeline.h>
#include <vtkSmartPointer.h>
#include <vtkDataObject.h>
#include <vtkLogger.h>

#include <vtkImageData.h>
#include <vtkPolyData.h>

vtkStandardNewMacro(prtlMyFilter);

prtlMyFilter::prtlMyFilter() {
  this->SetNumberOfInputPorts(1);
  this->SetNumberOfOutputPorts(1);
}

prtlMyFilter::~prtlMyFilter() = default;

int prtlMyFilter::FillInputPortInformation(int port, vtkInformation *info) {
  if (port == 0) {
    info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkImageData");
  }
  return 1;
}

int prtlMyFilter::FillOutputPortInformation(int port, vtkInformation *info) {
  if (port == 0) {
    info->Set(vtkDataObject::DATA_TYPE_NAME(), "vtkPolyData");
  }
  return 1;
}

int prtlMyFilter::RequestDataObject(
    vtkInformation * vtkNotUsed(request),
    vtkInformationVector ** vtkNotUsed(inputVector),
    vtkInformationVector *outputVector) {
  prtl::vtk::detail::default_request_data_object<vtkPolyData>(this, outputVector, 0);
  return 1;
}

int prtlMyFilter::RequestInformation(
    vtkInformation * vtkNotUsed(request),
    vtkInformationVector **inputVector,
    vtkInformationVector *outputVector) {
  return 1;
}

int prtlMyFilter::RequestUpdateExtent(
    vtkInformation * vtkNotUsed(request),
    vtkInformationVector **inputVector,
    vtkInformationVector *outputVector) {
  return 1;
}

int prtlMyFilter::RequestData(
    vtkInformation * vtkNotUsed(request),
    vtkInformationVector **inputVector,
    vtkInformationVector *outputVector) {
  vtkImageData *input = vtkImageData::GetData(inputVector[0], 0);
  vtkPolyData *output = vtkPolyData::GetData(outputVector, 0);

  if (!input || !output)
    return 1;

  return 1;
}
