#ifndef vtkBifurcationLine_h
#define vtkBifurcationLine_h

#include "BifurcationLineModuleModule.h"   // generated export macro

#include <vtkParallelVectors.h>
#include <vtkPolyDataAlgorithm.h>
#include <vtkSmartPointer.h>

/* ------------------------------------------------------------------------- */
/*  Internal helper – runs the actual Parallel-Vectors operator and computes
 *  κ = –(λ₁ λ₂).  Users never create this directly; the wrapper does.
 */
class BIFURCATIONLINEMODULE_EXPORT vtkParallelVectorsForBifurcationLine
  : public vtkParallelVectors
{
public:
  static vtkParallelVectorsForBifurcationLine* New();
  vtkTypeMacro(vtkParallelVectorsForBifurcationLine, vtkParallelVectors);

  void SetJacobianDataArray(vtkSmartPointer<vtkDataArray> arr) { this->Jacobian = arr; }

  vtkSetMacro(EnableThreshold, bool);
  vtkGetMacro(EnableThreshold, bool);
  vtkSetMacro(MinimumCriterion, double);
  vtkGetMacro(MinimumCriterion, double);
  vtkSetMacro(MaximumCriterion, double);
  vtkGetMacro(MaximumCriterion, double);

protected:
  vtkParallelVectorsForBifurcationLine();
  ~vtkParallelVectorsForBifurcationLine() override = default;

  void Prefilter(vtkInformation*, vtkInformationVector**, vtkInformationVector*) override;
  bool ComputeAdditionalCriteria(const vtkIdType triPts[3],
                                 double s, double t,
                                 std::vector<double>& vals) override;

  void Postfilter(vtkInformation*, vtkInformationVector**,
                  vtkInformationVector* outputVector) override;

  vtkSmartPointer<vtkDataArray> Jacobian;
  bool   EnableThreshold;
  double MinimumCriterion;
  double MaximumCriterion;

private:
  vtkParallelVectorsForBifurcationLine(const vtkParallelVectorsForBifurcationLine&) = delete;
  void operator=(const vtkParallelVectorsForBifurcationLine&) = delete;
};

/* ------------------------------------------------------------------------- */
/*  User-visible filter – lets the user pick the two vector arrays, builds
 *  Jacobian & acceleration, then runs the helper above.
 */
class BIFURCATIONLINEMODULE_EXPORT vtkBifurcationLine
  : public vtkPolyDataAlgorithm
{
public:
  static vtkBifurcationLine* New();
  vtkTypeMacro(vtkBifurcationLine, vtkPolyDataAlgorithm);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  vtkSetStringMacro(PrimaryVectorFieldName);
  vtkGetStringMacro(PrimaryVectorFieldName);
  vtkSetStringMacro(SecondaryVectorFieldName);
  vtkGetStringMacro(SecondaryVectorFieldName);

  vtkSetMacro(EnableThreshold, bool);
  vtkGetMacro(EnableThreshold, bool);
  vtkSetMacro(MinimumCriterion, double);
  vtkGetMacro(MinimumCriterion, double);
  vtkSetMacro(MaximumCriterion, double);
  vtkGetMacro(MaximumCriterion, double);

protected:
  vtkBifurcationLine();
  ~vtkBifurcationLine() override;

  int FillInputPortInformation(int, vtkInformation*) override;
  int RequestData(vtkInformation*, vtkInformationVector**,
                  vtkInformationVector*) override;

  char* PrimaryVectorFieldName;
  char* SecondaryVectorFieldName;

  bool   EnableThreshold;
  double MinimumCriterion;
  double MaximumCriterion;

private:
  vtkBifurcationLine(const vtkBifurcationLine&) = delete;
  void operator=(const vtkBifurcationLine&) = delete;
};

#endif
