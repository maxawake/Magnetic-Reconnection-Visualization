#include "vtkBifurcationLine.h"
#include <vtkObjectFactory.h>
#include <vtkDataSet.h>
#include <vtkPointData.h>
#include <vtkDoubleArray.h>
#include <vtkGradientFilter.h>
#include <vtkNew.h>
#include <vtkSmartPointer.h>
#include <vtkInformation.h>
#include <vtkInformationVector.h>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

vtkStandardNewMacro(vtkBifurcationLine);

vtkBifurcationLine::vtkBifurcationLine() {
  this->SetNumberOfInputPorts(1);
  this->SetNumberOfOutputPorts(1);
}

int vtkBifurcationLine::RequestData(vtkInformation* request,
                                    vtkInformationVector** inputVector,
                                    vtkInformationVector* outputVector)
{
  // Get input/output
  vtkInformation* inInfo = inputVector[0]->GetInformationObject(0);
  vtkDataSet* input = vtkDataSet::SafeDownCast(inInfo->Get(vtkDataObject::DATA_OBJECT()));
  vtkInformation* outInfo = outputVector->GetInformationObject(0);
  vtkPolyData* output = vtkPolyData::SafeDownCast(outInfo->Get(vtkDataObject::DATA_OBJECT()));

  if (!input || !output)
  {
    vtkErrorMacro("Invalid input or output");
    return 0;
  }

  // 1. Get vector field
  auto* vField = input->GetPointData()->GetVectors(this->FirstVectorFieldName);
  if (!vField)
  {
    vtkErrorMacro("Missing vector field: " << this->FirstVectorFieldName);
    return 0;
  }

  // 2. Compute gradient (∇v)
  vtkNew<vtkGradientFilter> gradient;
  gradient->SetInputData(input);
  gradient->SetInputArrayToProcess(0, 0, 0,
    vtkDataObject::FIELD_ASSOCIATION_POINTS, vField->GetName());
  gradient->SetResultArrayName("Jacobian");
  gradient->SetContainerAlgorithm(this);
  gradient->Update();

  vtkDataSet* withGradient = gradient->GetOutput();
  auto* jacobian = withGradient->GetPointData()->GetArray("Jacobian");
  if (!jacobian)
  {
    vtkErrorMacro("Failed to compute velocity gradient");
    return 0;
  }

  // 3. Compute acceleration: a = J · v
  vtkNew<vtkDoubleArray> acceleration;
  acceleration->SetName("Acceleration");
  acceleration->SetNumberOfComponents(3);
  acceleration->SetNumberOfTuples(vField->GetNumberOfTuples());

  // Simple J*v at each point
  for (vtkIdType i = 0; i < vField->GetNumberOfTuples(); ++i)
  {
    double J[9], v[3], a[3] = { 0, 0, 0 };
    jacobian->GetTuple(i, J);
    vField->GetTuple(i, v);

    for (int row = 0; row < 3; ++row)
      for (int col = 0; col < 3; ++col)
        a[row] += J[row * 3 + col] * v[col];

    acceleration->SetTuple(i, a);
  }

  withGradient->GetPointData()->AddArray(acceleration);
  withGradient->GetPointData()->SetActiveVectors(acceleration->GetName());

  // 4. Configure ParallelVectors operator
  this->SetInputData(withGradient);
  this->SetSecondVectorFieldName("Acceleration");

  // 5. Run vtkParallelVectors logic
  this->Superclass::RequestData(request, inputVector, outputVector);
  output->ShallowCopy(this->GetOutput());
  return 1;
}


bool vtkBifurcationLine::ComputeAdditionalCriteria(
  const vtkIdType* ids, double s, double t, std::vector<double>& criteria)
{
  vtkDataSet* input = vtkDataSet::SafeDownCast(this->GetInput());
  auto* field = input->GetPointData()->GetVectors(this->FirstVectorFieldName);

  double p[3][3], v[3][3];
  for (int i = 0; i < 3; ++i) {
    input->GetPoint(ids[i], p[i]);
    field->GetTuple(ids[i], v[i]);
  }

  // Interpolate vector field
  double w0 = 1.0 - s - t, w1 = s, w2 = t;
  double vec[3];
  for (int i = 0; i < 3; ++i)
    vec[i] = w0 * v[0][i] + w1 * v[1][i] + w2 * v[2][i];

  // Approximate Jacobian (2 vectors from points 1 and 2 - point 0)
  Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
  for (int i = 1; i < 3; ++i) {
    Eigen::Vector3d dx = Eigen::Map<Eigen::Vector3d>(p[i]) - Eigen::Map<Eigen::Vector3d>(p[0]);
    Eigen::Vector3d dv = Eigen::Map<Eigen::Vector3d>(v[i]) - Eigen::Map<Eigen::Vector3d>(v[0]);
    if (dx.norm() > 1e-10)
      J.col(i - 1) = dv / dx.norm();
  }

  // Eigenvalue computation
  Eigen::EigenSolver<Eigen::Matrix3d> solver(J);
  Eigen::Vector3cd evals = solver.eigenvalues();

  std::vector<double> realEvals;
  for (int i = 0; i < 3; ++i)
    if (std::abs(evals[i].imag()) < 1e-8)
      realEvals.push_back(evals[i].real());

  if (realEvals.size() < 2)
    return false;

  criteria = { -realEvals[0] * realEvals[1] };
  return true;
}

