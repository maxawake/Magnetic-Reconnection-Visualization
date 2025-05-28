#pragma once

#include <vtkDataObjectAlgorithm.h>
#include "prtlMyFilterModule.h"

/**
 * @brief 
 */
class PRTLMYFILTER_EXPORT prtlMyFilter : public vtkDataObjectAlgorithm {
 public:
  static prtlMyFilter *New();
  vtkTypeMacro(prtlMyFilter, vtkDataObjectAlgorithm);

 protected:
  prtlMyFilter();
  ~prtlMyFilter() override;

  int FillInputPortInformation(int port, vtkInformation* info) override;
  int FillOutputPortInformation(int port, vtkInformation* info) override;

  int RequestInformation(
      vtkInformation* request,
      vtkInformationVector** inputVector,
      vtkInformationVector* outputVector) override;

  int RequestUpdateExtent(
      vtkInformation* request,
      vtkInformationVector** inputVector,
      vtkInformationVector* outputVector) override;

  int RequestDataObject(
      vtkInformation* request,
      vtkInformationVector** inputVector,
      vtkInformationVector* outputVector) override;

  int RequestData(
      vtkInformation* request,
      vtkInformationVector** inputVector,
      vtkInformationVector* outputVector) override;

 private:
  prtlMyFilter(const prtlMyFilter&); // Not implemented.
  void operator=(const prtlMyFilter&); // Not implemented.
};
