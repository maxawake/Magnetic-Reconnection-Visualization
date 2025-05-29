#ifndef vtkBifurcationLine_H
#define vtkBifurcationLine_H

#include "BifurcationLineModuleModule.h"  // this must match LIBRARY_NAME + Module.h

#include <vtkPolyDataAlgorithm.h>
#include <vtkInformation.h>              // Required for vtkInformation
#include <vtkInformationVector.h>        // Required for vtkInformationVector

#include <vtkParallelVectors.h>

class BIFURCATIONLINEMODULE_EXPORT vtkBifurcationLine : public vtkParallelVectors
{

public:
  static vtkBifurcationLine* New();
  vtkTypeMacro(vtkBifurcationLine, vtkParallelVectors);

protected:
  vtkBifurcationLine();
  ~vtkBifurcationLine() override = default;

  int RequestData(vtkInformation*, vtkInformationVector**, vtkInformationVector*) override;
  bool ComputeAdditionalCriteria(const vtkIdType* triIds, double s, double t,
                                 std::vector<double>& criteria) override;

private:
  vtkBifurcationLine(const vtkBifurcationLine&) = delete;
  void operator=(const vtkBifurcationLine&) = delete;
};

#endif // vtkBifurcationLine_H
