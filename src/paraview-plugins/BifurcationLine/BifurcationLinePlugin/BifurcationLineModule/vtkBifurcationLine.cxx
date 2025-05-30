#include "vtkBifurcationLine.h"

#include "vtkArrayDispatch.h"
#include "vtkCharArray.h"
#include "vtkDataArray.h"
#include "vtkDataSet.h"
#include "vtkDoubleArray.h"
#include "vtkGradientFilter.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkNew.h"
#include "vtkObjectFactory.h"
#include "vtkPointData.h"
#include "vtkPolyData.h"
#include "vtkSMPTools.h"
#include <vtkCellData.h>


#include <eigen3/Eigen/Eigenvalues>
#include <eigen3/Eigen/Geometry>

#include <algorithm>
#include <array>

//-----------------------------------------------------------------------------
// Compute κ = –(λ_min * λ_max)
static bool computeBifurcationCriteria(const double J[9], double &criterion)
{
    Eigen::Matrix<double, 3, 3> mat;
    for (int i = 0; i < 9; ++i)
        mat(i / 3, i % 3) = J[i];
    Eigen::EigenSolver<Eigen::Matrix<double, 3, 3>> es(mat, false);
    std::array<double, 3> lam = {
        es.eigenvalues()[0].real(),
        es.eigenvalues()[1].real(),
        es.eigenvalues()[2].real()};
    std::sort(lam.begin(), lam.end()); // lam[0]=min, lam[2]=max
    criterion = -(lam[0] * lam[2]);
    return true;
}

//-----------------------------------------------------------------------------
// Functor from vtkVortexCore: A*b = x  (here a 3×3 jacobian × vector → acceleration)
template <typename AArrayType, typename BArrayType, typename XArrayType>
class MatrixVectorMultiplyFunctor
{
    AArrayType *AArray;
    BArrayType *BArray;
    XArrayType *XArray;
    vtkBifurcationLine *Filter;

public:
    MatrixVectorMultiplyFunctor(
        AArrayType *a, BArrayType *b, XArrayType *x, vtkBifurcationLine *f)
        : AArray(a), BArray(b), XArray(x), Filter(f) {}

    void operator()(vtkIdType begin, vtkIdType end)
    {
        const auto aRange = vtk::DataArrayTupleRange<9>(AArray, begin, end);
        const auto bRange = vtk::DataArrayTupleRange<3>(BArray, begin, end);
        auto xRange = vtk::DataArrayTupleRange<3>(XArray, begin, end);

        auto aIt = aRange.cbegin();
        auto bIt = bRange.cbegin();
        auto xIt = xRange.begin();
        bool first = vtkSMPTools::GetSingleThread();

        for (; aIt != aRange.cend(); ++aIt, ++bIt, ++xIt)
        {
            if (first)
            {
                Filter->CheckAbort();
            }
            if (Filter->GetAbortOutput())
            {
                break;
            }

            for (int i = 0; i < 3; ++i)
            {
                (*xIt)[i] =
                    (*aIt)[0 + i * 3] * (*bIt)[0] +
                    (*aIt)[1 + i * 3] * (*bIt)[1] +
                    (*aIt)[2 + i * 3] * (*bIt)[2];
            }
        }
    }
};

struct MatrixVectorMultiplyWorker
{
    template <typename AArrayType, typename BArrayType, typename XArrayType>
    void operator()(AArrayType *a, BArrayType *b, XArrayType *x, vtkBifurcationLine *f)
    {
        MatrixVectorMultiplyFunctor<AArrayType, BArrayType, XArrayType> fun(a, b, x, f);
        vtkSMPTools::For(0, x->GetNumberOfTuples(), fun);
    }
};

//-----------------------------------------------------------------------------
// Functor to build accepted‐points mask in parallel
struct ComputeMaskFunctor
{
    vtkDataArray *Jacobian;
    vtkCharArray *Mask;
    double MinC, MaxC;
    vtkBifurcationLine *Filter;

    ComputeMaskFunctor(vtkDataArray *j, vtkCharArray *m,
                       double minC, double maxC,
                       vtkBifurcationLine *f)
        : Jacobian(j), Mask(m), MinC(minC), MaxC(maxC), Filter(f) {}

    void operator()(vtkIdType begin, vtkIdType end)
    {
        double J[9], crit;
        for (vtkIdType i = begin; i < end; ++i)
        {
            if (Filter->CheckAbort() || Filter->GetAbortOutput())
                break;
            Jacobian->GetTuple(i, J);
            computeBifurcationCriteria(J, crit);
            bool keep = !Filter->GetEnableThreshold() ||
                        (crit >= MinC && crit <= MaxC);
            Mask->SetValue(i, keep ? 1 : 0);
        }
    }
};

//-----------------------------------------------------------------------------
// vtkParallelVectorsForBifurcationLine
vtkStandardNewMacro(vtkParallelVectorsForBifurcationLine);

void vtkParallelVectorsForBifurcationLine::Prefilter(
    vtkInformation *, vtkInformationVector **, vtkInformationVector *)
{
    this->CriteriaArrays.resize(1);
    vtkNew<vtkDoubleArray> arr;
    arr->SetName("bifurcation_criterion");
    this->CriteriaArrays[0] = arr;
}

bool vtkParallelVectorsForBifurcationLine::AcceptSurfaceTriangle(
    const vtkIdType pts[3])
{
    auto mask = this->AcceptedPoints->GetPointer(0);
    return mask[pts[0]] && mask[pts[1]] && mask[pts[2]];
}

bool vtkParallelVectorsForBifurcationLine::ComputeAdditionalCriteria(
    const vtkIdType pts[3], double s, double t,
    std::vector<double> &values)
{
    double Jv[3][9], Ji[9], crit;
    for (int i = 0; i < 3; ++i)
    {
        this->Jacobian->GetTuple(pts[i], Jv[i]);
    }
    for (int k = 0; k < 9; ++k)
        Ji[k] = (1. - s - t) * Jv[0][k] + s * Jv[1][k] + t * Jv[2][k];
    computeBifurcationCriteria(Ji, crit);
    values[0] = crit;
    return true;
}

//-----------------------------------------------------------------------------
// vtkBifurcationLine
vtkStandardNewMacro(vtkBifurcationLine);

vtkBifurcationLine::vtkBifurcationLine()
    : EnableThreshold(false), MinimumCriterion(0.0), MaximumCriterion(VTK_DOUBLE_MAX)
{
    this->SetNumberOfInputPorts(1);
    this->SetNumberOfOutputPorts(1);
}

vtkBifurcationLine::~vtkBifurcationLine() = default;

int vtkBifurcationLine::FillInputPortInformation(int, vtkInformation *info)
{
    info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkDataSet");
    return 1;
}

int vtkBifurcationLine::RequestData(
    vtkInformation *,
    vtkInformationVector **inVec,
    vtkInformationVector *outVec)
{
    // fetch input velocity
    vtkDataSet *input = vtkDataSet::GetData(inVec[0]);
    vtkDataArray *velocity =
        input->GetPointData()->GetVectors(this->PrimaryVectorFieldName);
    if (!velocity)
    {
        vtkErrorMacro("PrimaryVectorFieldName not set or not a 3-component vector");
        return 0;
    }

    // compute ∇v (Jacobian)
    vtkNew<vtkGradientFilter> grad;
    grad->SetInputData(input);
    grad->SetResultArrayName("Jacobian");
    grad->ComputeVorticityOff();
    grad->SetInputArrayToProcess(
        0, 0, 0,
        vtkDataObject::FIELD_ASSOCIATION_POINTS,
        velocity->GetName());
    grad->Update();
    vtkDataSet *gradOut = grad->GetOutput();
    vtkDataArray *jacobian =
        gradOut->GetPointData()->GetArray("Jacobian");

    // acceleration = J * v  (multi-threaded)
    vtkNew<vtkDoubleArray> acceleration;
    acceleration->SetName("acceleration");
    acceleration->SetNumberOfComponents(3);
    vtkIdType nt = velocity->GetNumberOfTuples();
    acceleration->SetNumberOfTuples(nt);

    MatrixVectorMultiplyWorker mvWorker;
    using Dispatcher = vtkArrayDispatch::Dispatch3ByValueType<
        vtkArrayDispatch::Reals,
        vtkArrayDispatch::Reals,
        vtkArrayDispatch::Reals>;
    if (!Dispatcher::Execute(jacobian, velocity,
                             acceleration.GetPointer(),
                             mvWorker, this))
    {
        mvWorker(jacobian, velocity,
                 acceleration.GetPointer(), this);
    }
    gradOut->GetPointData()->AddArray(acceleration);

    // build accepted-points mask (multi-threaded)
    vtkNew<vtkCharArray> acceptedPoints;
    acceptedPoints->SetName("acceptedPoints");
    acceptedPoints->SetNumberOfTuples(nt);

    ComputeMaskFunctor maskFun(jacobian,
                               acceptedPoints.GetPointer(),
                               this->MinimumCriterion,
                               this->MaximumCriterion,
                               this);
    vtkSMPTools::For(0, nt, maskFun);

    // run parallel-vectors
    vtkNew<vtkParallelVectorsForBifurcationLine> pv;
    pv->SetInputData(gradOut);
    pv->SetJacobian(jacobian);
    pv->SetAcceptedPoints(acceptedPoints.GetPointer());
    pv->SetFirstVectorFieldName(velocity->GetName());
    pv->SetSecondVectorFieldName(acceleration->GetName());
    pv->Update();

    vtkPolyData *raw = pv->GetOutput();
    vtkNew<vtkPolyData> filtered;
    vtkNew<vtkPoints> newPoints;
    vtkNew<vtkCellArray> newLines;

    // we'll need to copy point‐data arrays
    filtered->GetPointData()->ShallowCopy(raw->GetPointData());

    std::unordered_map<vtkIdType, vtkIdType> pointMap;
    vtkIdType oldId, newId;
    const double tol = 1e-6;
    vtkIdType npts;
    const vtkIdType *pts;

    // iterate each line cell
    raw->GetLines()->InitTraversal();
    while (raw->GetLines()->GetNextCell(npts, pts))
    {
        // length filter (#cells = npts-1)
        if (this->EnableLengthFilter && (npts - 1) < this->MinimumCells)
        {
            continue;
        }

        // angle‐turn filter
        bool badTurn = false;
        if (this->EnableAngleFilter && npts >= 3)
        {
            for (vtkIdType i = 0; i + 2 < npts; ++i)
            {
                double p0[3], p1[3], p2[3], v1[3], v2[3];
                raw->GetPoint(pts[i], p0);
                raw->GetPoint(pts[i + 1], p1);
                raw->GetPoint(pts[i + 2], p2);
                vtkMath::Subtract(p1, p0, v1);
                vtkMath::Subtract(p2, p1, v2);
                vtkMath::Normalize(v1);
                vtkMath::Normalize(v2);
                double dot = std::clamp(vtkMath::Dot(v1, v2), -1.0, 1.0);
                double angle = vtkMath::DegreesFromRadians(acos(dot));
                if (angle > this->MaximumTangentAngle)
                {
                    badTurn = true;
                    break;
                }
            }
        }
        if (badTurn)
        {
            continue;
        }

        // accept this line → remap its points
        std::vector<vtkIdType> newIds(npts);
        for (vtkIdType j = 0; j < npts; ++j)
        {
            oldId = pts[j];
            auto it = pointMap.find(oldId);
            if (it == pointMap.end())
            {
                double p[3];
                raw->GetPoint(oldId, p);
                newId = newPoints->InsertNextPoint(p);
                pointMap[oldId] = newId;
                it = pointMap.find(oldId);
            }
            newIds[j] = it->second;
        }
        newLines->InsertNextCell(npts, newIds.data());
    }

    // plug into filtered output
    filtered->SetPoints(newPoints);
    filtered->SetLines(newLines);

    // finally shallow‐copy any remaining cell‐data arrays
    filtered->GetCellData()->ShallowCopy(raw->GetCellData());

    // hand it back
    vtkPolyData *outPd = vtkPolyData::GetData(outVec, 0);
    outPd->ShallowCopy(filtered);
    return 1;
}

void vtkBifurcationLine::PrintSelf(ostream &os, vtkIndent indent)
{
    this->Superclass::PrintSelf(os, indent);
    os << indent << "PrimaryVectorFieldName: "
       << (this->PrimaryVectorFieldName ? this->PrimaryVectorFieldName : "(none)") << "\n";
    os << indent << "EnableThreshold: "
       << (this->EnableThreshold ? "On" : "Off") << "\n";
    os << indent << "MinimumCriterion: " << this->MinimumCriterion << "\n";
    os << indent << "MaximumCriterion: " << this->MaximumCriterion << "\n";
}
