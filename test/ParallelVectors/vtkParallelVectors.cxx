#include "vtkParallelVectors.h"
#include <vtkInformation.h>
#include <vtkInformationVector.h>
#include <vtkObjectFactory.h>
#include <vtkPolyData.h>

vtkStandardNewMacro(vtkParallelVectors);

vtkParallelVectors::vtkParallelVectors()
{
  this->SetNumberOfInputPorts(1);
  this->SetNumberOfOutputPorts(1);
}

int vtkParallelVectors::RequestData(vtkInformation*,
                                    vtkInformationVector**,
                                    vtkInformationVector*)
{
  // dummy – returns empty polydata
  return 1;
}
