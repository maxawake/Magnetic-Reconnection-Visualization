#ifndef vtkBifurcationLineMSR_h
#define vtkBifurcationLineMSR_h

#include "vtkParallelVectors.h"
#include "vtkFiltersFlowPathsModule.h"   // export macro

VTK_ABI_NAMESPACE_BEGIN
class VTKFILTERSFLOWPATHS_EXPORT vtkBifurcationLineMSR :
  public vtkParallelVectors
{
public:
  enum CriterionType { VKA = 0, VKB = 1 };

  static vtkBifurcationLineMSR* New();
  vtkTypeMacro(vtkBifurcationLineMSR, vtkParallelVectors);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  /// `VKA` (velocity ‖ acceleration) or `VKB` (velocity ‖ jerk)
  vtkSetClampMacro(Criterion, int, 0, 1);
  vtkGetMacro(Criterion, int);

  /// angle filter α(ϑ,v) ≤ AngleThreshold  [deg]  (cf. [PR99])
  vtkSetClampMacro(AngleThreshold, double,   0.0, 90.0);
  vtkGetMacro   (AngleThreshold, double);

  /// hyperbolicity filter  det(J⊥)/‖v‖ < −HyperbolicityThreshold
  vtkSetMacro(HyperbolicityThreshold, double);
  vtkGetMacro(HyperbolicityThreshold, double);

protected:
  vtkBifurcationLineMSR();
  ~vtkBifurcationLineMSR() override = default;

  // pipeline customisation --------------------------------------------------
  void Prefilter(vtkInformation*, vtkInformationVector**,
                 vtkInformationVector*) override;

  bool ComputeAdditionalCriteria(const vtkIdType triPtIds[3],
                                 double s, double t,
                                 std::vector<double>& criteria) override;

  void Postfilter(vtkInformation*, vtkInformationVector**,
                  vtkInformationVector*) override;

  int  FillInputPortInformation(int, vtkInformation*) override;

private:
  vtkBifurcationLineMSR(const vtkBifurcationLineMSR&) = delete;
  void operator=(const vtkBifurcationLineMSR&)       = delete;

  int    Criterion { VKA };
  double AngleThreshold { 30.0 };            // degrees
  double HyperbolicityThreshold { 1e-3 };    // Machado et al.

  std::vector<vtkSmartPointer<vtkDataArray>> Diagnostics;
};
VTK_ABI_NAMESPACE_END
#endif
