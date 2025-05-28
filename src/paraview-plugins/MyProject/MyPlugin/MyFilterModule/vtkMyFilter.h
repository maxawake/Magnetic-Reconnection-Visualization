#ifndef vtkMyFilter_H
#define vtkMyFilter_H

#include "MyFilterModulesModule.h"  // this must match LIBRARY_NAME + Module.h

#include <vtkPolyDataAlgorithm.h>
#include <vtkInformation.h>              // Required for vtkInformation
#include <vtkInformationVector.h>        // Required for vtkInformationVector

class MYFILTERMODULES_EXPORT vtkMyFilter : public vtkPolyDataAlgorithm {
public:
  static vtkMyFilter* New();
  vtkTypeMacro(vtkMyFilter, vtkPolyDataAlgorithm);

protected:
  vtkMyFilter();
  ~vtkMyFilter() override = default;

  int RequestData(vtkInformation*, vtkInformationVector**, vtkInformationVector*) override;

private:
  vtkMyFilter(const vtkMyFilter&) = delete;
  void operator=(const vtkMyFilter&) = delete;
};

#endif // vtkMyFilter_H
