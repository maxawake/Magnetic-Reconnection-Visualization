#include "vtkBifurcationLineMSR.h"

#include "vtkArrayCalculator.h"
#include "vtkGradientFilter.h"
#include "vtkMath.h"
#include "vtkNew.h"
#include "vtkObjectFactory.h"
#include "vtkPointData.h"
#include "vtkTriangle.h"
#include "vtkDataSet.h"
#include "vtkInformationVector.h"
#include "vtkInformation.h"
#include "vtkExecutive.h"
#include "vtkDoubleArray.h"
#include "vtkSmartPointer.h"

vtkStandardNewMacro(vtkBifurcationLineMSR);

vtkBifurcationLineMSR::vtkBifurcationLineMSR()
{
  this->SetFirstVectorFieldName("Velocity");

  vtkNew<vtkDoubleArray> detPerp;
  detPerp->SetName("det_perp");
  this->Diagnostics.push_back(detPerp);
}

void vtkBifurcationLineMSR::Prefilter(
  vtkInformation*,
  vtkInformationVector** inVec,
  vtkInformationVector*)
{
  auto* inputDO = inVec[0]->GetInformationObject(0)->Get(vtkDataObject::DATA_OBJECT());
  auto* ds = vtkDataSet::SafeDownCast(inputDO);

  vtkNew<vtkGradientFilter> gV;
  gV->SetInputData(ds);
  gV->SetInputScalars(vtkDataObject::FIELD_ASSOCIATION_POINTS,
                      this->FirstVectorFieldName);
  gV->SetResultArrayName("GradV");
  gV->Update();

  vtkNew<vtkArrayCalculator> calcA;
  calcA->SetInputConnection(gV->GetOutputPort());
  calcA->AddVectorArrayName(this->FirstVectorFieldName);
  calcA->AddVectorArrayName("GradV");
  calcA->SetResultArrayName("Acceleration");
  calcA->SetFunction(
    "iHat*(Velocity_X*GradV_XX + Velocity_Y*GradV_YX + Velocity_Z*GradV_ZX)"
    "+jHat*(Velocity_X*GradV_XY + Velocity_Y*GradV_YY + Velocity_Z*GradV_ZY)"
    "+kHat*(Velocity_X*GradV_XZ + Velocity_Y*GradV_YZ + Velocity_Z*GradV_ZZ)");
  calcA->Update();

  vtkAlgorithmOutput* current = calcA->GetOutputPort();

  if (this->Criterion == VKB)
  {
    vtkNew<vtkGradientFilter> gA;
    gA->SetInputConnection(current);
    gA->SetInputScalars(vtkDataObject::FIELD_ASSOCIATION_POINTS, "Acceleration");
    gA->SetResultArrayName("GradA");
    gA->Update();

    vtkNew<vtkArrayCalculator> calcB;
    calcB->SetInputConnection(gA->GetOutputPort());
    calcB->AddVectorArrayName(this->FirstVectorFieldName);
    calcB->AddVectorArrayName("GradA");
    calcB->SetResultArrayName("Jerk");
    calcB->SetFunction(
      "iHat*(Velocity_X*GradA_XX + Velocity_Y*GradA_YX + Velocity_Z*GradA_ZX)"
      "+jHat*(Velocity_X*GradA_XY + Velocity_Y*GradA_YY + Velocity_Z*GradA_ZY)"
      "+kHat*(Velocity_X*GradA_XZ + Velocity_Y*GradA_YZ + Velocity_Z*GradA_ZZ)");
    calcB->Update();
    current = calcB->GetOutputPort();
  }

  this->SetInputConnection(0, current);

  if (this->Criterion == VKA)
    this->SetSecondVectorFieldName("Acceleration");
  else
    this->SetSecondVectorFieldName("Jerk");

  vtkDataSet* result = vtkDataSet::SafeDownCast(this->GetInput());
  if (result)
  {
    vtkDataArray* gradArray = gV->GetOutput()->GetPointData()->GetArray("GradV");
    if (gradArray)
    {
      result->GetPointData()->AddArray(gradArray);
    }
  }
}

bool vtkBifurcationLineMSR::ComputeAdditionalCriteria(
  const vtkIdType triPtIds[3], double s, double t,
  std::vector<double>& criteria)
{
  double w0 = 1.0 - s - t;
  double w1 = s;
  double w2 = t;

  vtkDataSet* in = vtkDataSet::SafeDownCast(this->GetInput());
  if (!in)
    return false;

  vtkDataArray* vel = in->GetPointData()->GetArray(this->FirstVectorFieldName);
  vtkDataArray* gV  = in->GetPointData()->GetArray("GradV");

  double v[3];
  for (int i = 0; i < 3; ++i)
  {
    v[i] = w0 * vel->GetComponent(triPtIds[0], i) +
           w1 * vel->GetComponent(triPtIds[1], i) +
           w2 * vel->GetComponent(triPtIds[2], i);
  }

  double vMag = vtkMath::Norm(v);
  if (vMag < VTK_DBL_EPSILON)
    return false;

  double vHat[3] = {v[0] / vMag, v[1] / vMag, v[2] / vMag};

  double A[3][3];
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c)
    {
      int comp = 3 * r + c;
      A[r][c] = w0 * gV->GetComponent(triPtIds[0], comp)
              + w1 * gV->GetComponent(triPtIds[1], comp)
              + w2 * gV->GetComponent(triPtIds[2], comp);
    }

  double e1[3], e2[3];
  {
    double w[3] = {0, 0, 0};
    int minIdx = (std::abs(vHat[0]) < std::abs(vHat[1])) ?
                   ((std::abs(vHat[0]) < std::abs(vHat[2])) ? 0 : 2) :
                   ((std::abs(vHat[1]) < std::abs(vHat[2])) ? 1 : 2);
    w[minIdx] = 1.0;
    vtkMath::Cross(vHat, w, e1);
    vtkMath::Normalize(e1);
    vtkMath::Cross(vHat, e1, e2);
    vtkMath::Normalize(e2);
  }

  auto dot = [](const double* a, const double* b)
  {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
  };

  double col0[3] = {
    A[0][0]*e1[0] + A[1][0]*e1[1] + A[2][0]*e1[2],
    A[0][1]*e1[0] + A[1][1]*e1[1] + A[2][1]*e1[2],
    A[0][2]*e1[0] + A[1][2]*e1[1] + A[2][2]*e1[2]
  };

  double col1[3] = {
    A[0][0]*e2[0] + A[1][0]*e2[1] + A[2][0]*e2[2],
    A[0][1]*e2[0] + A[1][1]*e2[1] + A[2][1]*e2[2],
    A[0][2]*e2[0] + A[1][2]*e2[1] + A[2][2]*e2[2]
  };

  double m00 = dot(e1, col0);
  double m01 = dot(e1, col1);
  double m10 = dot(e2, col0);
  double m11 = dot(e2, col1);

  double detPerp = m00 * m11 - m01 * m10;
  double detNorm = detPerp / vMag;

  criteria.resize(1);
  criteria[0] = detNorm;

  return detNorm < -this->HyperbolicityThreshold;
}

void vtkBifurcationLineMSR::Postfilter(
  vtkInformation* req,
  vtkInformationVector** inVec,
  vtkInformationVector* outVec)
{
  this->Superclass::Postfilter(req, inVec, outVec);

  vtkPolyData* out = vtkPolyData::SafeDownCast(
    outVec->GetInformationObject(0)->Get(vtkDataObject::DATA_OBJECT()));

  for (auto& arr : this->Diagnostics)
    if (arr && arr->GetNumberOfTuples())
      out->GetPointData()->AddArray(arr);
}

int vtkBifurcationLineMSR::FillInputPortInformation(int port, vtkInformation* info)
{
  return this->Superclass::FillInputPortInformation(port, info);
}

void vtkBifurcationLineMSR::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  os << indent << "Criterion: "
     << (this->Criterion == VKA ? "VKA" : "VKB") << "\n";
  os << indent << "AngleThreshold: " << this->AngleThreshold << "\n";
  os << indent << "HyperbolicityThreshold: " << this->HyperbolicityThreshold << "\n";
}
