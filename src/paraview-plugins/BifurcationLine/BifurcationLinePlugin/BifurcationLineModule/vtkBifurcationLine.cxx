// vtkBifurcationLine.cxx

#include "vtkBifurcationLine.h"

#include <vtkArrayCalculator.h>
#include <vtkArrayDispatch.h>
#include <vtkCharArray.h>
#include <vtkDataArray.h>
#include <vtkDataSet.h>
#include <vtkDoubleArray.h>
#include <vtkGradientFilter.h>
#include <vtkInformation.h>
#include <vtkInformationVector.h>
#include <vtkNew.h>
#include <vtkObjectFactory.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkSMPTools.h>

#include <eigen3/Eigen/Eigenvalues>
#include <eigen3/Eigen/Geometry>

#include <algorithm>
#include <array>

// Compute κ = –(λ_min * λ_max)
static bool computeBifurcationCriteria(const double J[9], double &criterion)
{
  Eigen::Matrix<double,3,3> mat;
  for (int i = 0; i < 9; ++i)
  {
    mat(i/3, i%3) = J[i];
  }
  Eigen::EigenSolver<Eigen::Matrix<double,3,3>> es(mat, false);
  std::array<double,3> lam = {
    es.eigenvalues()[0].real(),
    es.eigenvalues()[1].real(),
    es.eigenvalues()[2].real()
  };
  std::sort(lam.begin(), lam.end()); // lam[0]=min, lam[2]=max
  criterion = -(lam[0] * lam[2]);
  return true;
}

//------------------------------------------------------------------------------
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
    double Jv[3][9];
    for (int i = 0; i < 3; ++i)
    {
        this->Jacobian->GetTuple(pts[i], Jv[i]);
    }
    double Ji[9];
    for (int k = 0; k < 9; ++k)
    {
        Ji[k] = (1. - s - t) * Jv[0][k] + s * Jv[1][k] + t * Jv[2][k];
    }
    double crit;
    computeBifurcationCriteria(Ji, crit);
    values[0] = crit;
    return true;
}

//------------------------------------------------------------------------------
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
    vtkInformation *, vtkInformationVector **inVec, vtkInformationVector *outVec)
{
    // 1) fetch input velocity
    vtkDataSet *input = vtkDataSet::GetData(inVec[0]);
    vtkDataArray *velocity =
        input->GetPointData()->GetVectors(this->PrimaryVectorFieldName);
    if (!velocity)
    {
        vtkErrorMacro("PrimaryVectorFieldName not set or not a 3-component vector");
        return 0;
    }

    // 2) compute Jacobian = ∇v
    vtkNew<vtkGradientFilter> grad;
    grad->SetInputData(input);
    grad->SetResultArrayName("Jacobian");
    grad->ComputeVorticityOff();
    grad->SetInputArrayToProcess(
        0, 0, 0, vtkDataObject::FIELD_ASSOCIATION_POINTS, velocity->GetName());
    grad->Update();
    vtkDataSet *gradOut = grad->GetOutput();
    vtkDataArray *jacobian =
        gradOut->GetPointData()->GetArray("Jacobian");

    // 3) compute acceleration a = J * v
    vtkNew<vtkDoubleArray> acceleration;
    acceleration->SetName("acceleration");
    acceleration->SetNumberOfComponents(3);
    vtkIdType nt = velocity->GetNumberOfTuples();
    acceleration->SetNumberOfTuples(nt);
    for (vtkIdType i = 0; i < nt; ++i)
    {
        double Jt[9], vv[3], aa[3] = {0, 0, 0};
        jacobian->GetTuple(i, Jt);
        velocity->GetTuple(i, vv);
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                aa[r] += Jt[3 * r + c] * vv[c];
        acceleration->SetTuple(i, aa);
    }
    gradOut->GetPointData()->AddArray(acceleration);

    // 4) build mask of accepted points
    vtkNew<vtkCharArray> acceptedPoints;
    acceptedPoints->SetName("acceptedPoints");
    acceptedPoints->SetNumberOfTuples(nt);
    for (vtkIdType i = 0; i < nt; ++i)
    {
        double Jt[9], crit;
        jacobian->GetTuple(i, Jt);
        computeBifurcationCriteria(Jt, crit);
        bool keep = !this->EnableThreshold || (crit >= this->MinimumCriterion && crit <= this->MaximumCriterion);
        acceptedPoints->SetValue(i, keep ? 1 : 0);
    }

    // 5) run parallel-vectors helper
    vtkNew<vtkParallelVectorsForBifurcationLine> pv;
    pv->SetInputData(gradOut);
    pv->SetJacobian(jacobian);
    pv->SetAcceptedPoints(acceptedPoints);
    pv->SetFirstVectorFieldName(velocity->GetName());
    pv->SetSecondVectorFieldName(acceleration->GetName());
    pv->Update();

    // 6) deliver output
    vtkPolyData *outPd = vtkPolyData::GetData(outVec, 0);
    outPd->ShallowCopy(pv->GetOutput());
    return 1;
}

void vtkBifurcationLine::PrintSelf(ostream &os, vtkIndent indent)
{
    this->Superclass::PrintSelf(os, indent);
    os << indent << "PrimaryVectorFieldName: "
       << (this->PrimaryVectorFieldName ? this->PrimaryVectorFieldName : "(none)") << "\n";
    os << indent << "SecondaryVectorFieldName: "
       << (this->SecondaryVectorFieldName ? this->SecondaryVectorFieldName : "(none)") << "\n";
    os << indent << "EnableThreshold: " << (this->EnableThreshold ? "On" : "Off") << "\n";
    os << indent << "MinimumCriterion: " << this->MinimumCriterion << "\n";
    os << indent << "MaximumCriterion: " << this->MaximumCriterion << "\n";
}
