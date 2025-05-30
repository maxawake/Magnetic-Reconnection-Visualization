// vtkBifurcationLine.h

#ifndef vtkBifurcationLine_h
#define vtkBifurcationLine_h

#include "BifurcationLineModuleModule.h" 
#include <vtkParallelVectors.h>
#include <vtkPolyDataAlgorithm.h>
#include <vtkSmartPointer.h>

#include <vtkCharArray.h>
#include <vtkDataArray.h>
/*- */
/*  Internal helper – runs the actual Parallel-Vectors operator and computes
 *  κ = –(λ_min * λ_max).  Users never create this directly; the wrapper does.
 */
class BIFURCATIONLINEMODULE_EXPORT vtkParallelVectorsForBifurcationLine
    : public vtkParallelVectors
{
public:
    static vtkParallelVectorsForBifurcationLine *New();
    vtkTypeMacro(vtkParallelVectorsForBifurcationLine, vtkParallelVectors);

    void SetJacobian(vtkDataArray *jac) { this->Jacobian = jac; }
    void SetAcceptedPoints(vtkCharArray *acc) { this->AcceptedPoints = acc; }

protected:
    vtkParallelVectorsForBifurcationLine() = default;
    ~vtkParallelVectorsForBifurcationLine() override = default;

    void Prefilter(vtkInformation *, vtkInformationVector **, vtkInformationVector *);
    bool AcceptSurfaceTriangle(const vtkIdType surfaceSimplexIndices[3]);
    bool ComputeAdditionalCriteria(const vtkIdType surfaceSimplexIndices[3],
                                   double s, double t,
                                   std::vector<double> &criterionArrayValues);

    vtkSmartPointer<vtkCharArray> AcceptedPoints;
    vtkSmartPointer<vtkDataArray> Jacobian;

private:
    vtkParallelVectorsForBifurcationLine(const vtkParallelVectorsForBifurcationLine &) = delete;
    void operator=(const vtkParallelVectorsForBifurcationLine &) = delete;
};

/*- */
/*  User-visible filter – lets the user pick the vector arrays, builds
 *  Jacobian & acceleration, then runs the helper above.
 */
class BIFURCATIONLINEMODULE_EXPORT vtkBifurcationLine
    : public vtkPolyDataAlgorithm
{
public:
    static vtkBifurcationLine *New();
    vtkTypeMacro(vtkBifurcationLine, vtkPolyDataAlgorithm);
    void PrintSelf(ostream &os, vtkIndent indent) override;

    vtkSetStringMacro(PrimaryVectorFieldName);
    vtkGetStringMacro(PrimaryVectorFieldName);
    vtkSetStringMacro(SecondaryVectorFieldName);
    vtkGetStringMacro(SecondaryVectorFieldName);

    // bifurcation criterion
    vtkSetMacro(EnableThreshold, bool);
    vtkGetMacro(EnableThreshold, bool);
    vtkSetMacro(MinimumCriterion, double);
    vtkGetMacro(MinimumCriterion, double);
    vtkSetMacro(MaximumCriterion, double);
    vtkGetMacro(MaximumCriterion, double);

    // length filter
    vtkSetMacro(EnableLengthFilter, bool);
    vtkGetMacro(EnableLengthFilter, bool);
    vtkSetMacro(MinimumCells, int);
    vtkGetMacro(MinimumCells, int);

    // angle‐turn filter
    vtkSetMacro(EnableAngleFilter, bool);
    vtkGetMacro(EnableAngleFilter, bool);
    vtkSetMacro(MaximumTangentAngle, double);
    vtkGetMacro(MaximumTangentAngle, double);

protected:
    vtkBifurcationLine();
    ~vtkBifurcationLine() override;

    int FillInputPortInformation(int, vtkInformation *) override;
    int RequestData(vtkInformation *, vtkInformationVector **, vtkInformationVector *) override;

    char *PrimaryVectorFieldName{nullptr};
    char *SecondaryVectorFieldName{nullptr};

    bool EnableThreshold;
    double MinimumCriterion;
    double MaximumCriterion;
    bool EnableLengthFilter;
    int MinimumCells;
    bool EnableAngleFilter;
    double MaximumTangentAngle;

private:
    vtkBifurcationLine(const vtkBifurcationLine &) = delete;
    void operator=(const vtkBifurcationLine &) = delete;
};

#endif // vtkBifurcationLine_h
