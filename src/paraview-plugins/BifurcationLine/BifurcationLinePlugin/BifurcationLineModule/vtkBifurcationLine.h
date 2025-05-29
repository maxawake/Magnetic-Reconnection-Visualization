#ifndef vtkBifurcationLine_H
#define vtkBifurcationLine_H

#include "BifurcationLineModuleModule.h"  // this must match LIBRARY_NAME + Module.h

#include <vtkPolyDataAlgorithm.h>
#include <vtkInformation.h>              // Required for vtkInformation
#include <vtkInformationVector.h>        // Required for vtkInformationVector

class BIFURCATIONLINEMODULE_EXPORT vtkBifurcationLine : public vtkPolyDataAlgorithm {
public:
  static vtkBifurcationLine* New();
  vtkTypeMacro(vtkBifurcationLine, vtkPolyDataAlgorithm);

protected:
  vtkBifurcationLine();
  ~vtkBifurcationLine() override = default;

  int RequestData(vtkInformation*, vtkInformationVector**, vtkInformationVector*) override;

private:
  vtkBifurcationLine(const vtkBifurcationLine&) = delete;
  void operator=(const vtkBifurcationLine&) = delete;
};

#endif // vtkBifurcationLine_H
