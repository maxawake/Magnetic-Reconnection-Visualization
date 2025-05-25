#ifndef vtkParallelVectors_h
#define vtkParallelVectors_h

#include <vtkPolyDataAlgorithm.h>

class vtkParallelVectors : public vtkPolyDataAlgorithm
{
public:
  static vtkParallelVectors* New();
  vtkTypeMacro(vtkParallelVectors, vtkPolyDataAlgorithm);

  vtkSetStringMacro(FirstVectorFieldName);
  vtkSetStringMacro(SecondVectorFieldName);

protected:
  vtkParallelVectors();
  int RequestData(vtkInformation*, vtkInformationVector**, vtkInformationVector*) override;

  char* FirstVectorFieldName = nullptr;
  char* SecondVectorFieldName = nullptr;
};

#endif
