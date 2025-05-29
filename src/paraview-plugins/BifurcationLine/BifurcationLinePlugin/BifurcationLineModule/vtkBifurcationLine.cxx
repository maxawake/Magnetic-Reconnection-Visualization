#include "vtkBifurcationLine.h"

#include <vtkCharArray.h>
#include <vtkDataArray.h>
#include <vtkDoubleArray.h>
#include <vtkGradientFilter.h>
#include <vtkInformation.h>
#include <vtkInformationVector.h>
#include <vtkNew.h>
#include <vtkObjectFactory.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <array>

#include <vtkCellLocator.h>
#include <vtkGenericCell.h>
#include <vtkProbeFilter.h>

/* ======================================================================= */
/*  Implementation – vtkParallelVectorsForBifurcationLine                  */
/* ======================================================================= */

vtkStandardNewMacro(vtkParallelVectorsForBifurcationLine);

//-----------------------------------------------------------------------------
vtkParallelVectorsForBifurcationLine::vtkParallelVectorsForBifurcationLine()
    : EnableThreshold(false), MinimumCriterion(0.0), MaximumCriterion(VTK_DOUBLE_MAX)
{
}

//-----------------------------------------------------------------------------
void vtkParallelVectorsForBifurcationLine::Prefilter(
    vtkInformation *, vtkInformationVector **, vtkInformationVector *)
{
    this->CriteriaArrays.resize(1);
    this->CriteriaArrays[0] = vtkSmartPointer<vtkDoubleArray>::New();
    this->CriteriaArrays[0]->SetName("bifurcation_criterion");
}

//-----------------------------------------------------------------------------
bool vtkParallelVectorsForBifurcationLine::ComputeAdditionalCriteria(
    const vtkIdType triPts[3], double s, double t,
    std::vector<double> &values)
{
    if (!this->Jacobian)
    {
        vtkErrorMacro("Jacobian array not set!");
        return false;
    }

    double Jv[3][9];
    for (int i = 0; i < 3; ++i)
        this->Jacobian->GetTuple(triPts[i], Jv[i]);

    Eigen::Matrix<double, 3, 3> J;
    for (int k = 0; k < 9; ++k)
        J(k / 3, k % 3) = (1. - s - t) * Jv[0][k] + s * Jv[1][k] + t * Jv[2][k];

    Eigen::EigenSolver<Eigen::Matrix<double, 3, 3>> es(J, false);
    std::array<double, 3> λ = {es.eigenvalues()[0].real(),
                               es.eigenvalues()[1].real(),
                               es.eigenvalues()[2].real()};
    std::sort(λ.begin(), λ.end(), std::greater<double>());

    double κ = -(λ[0] * λ[2]);

    if (this->EnableThreshold &&
        (κ < this->MinimumCriterion || κ > this->MaximumCriterion))
        return false;

    values[0] = κ;
    return true;
}

/* ------------------------------------------------------------------ */
/*  vtkParallelVectorsForBifurcationLine::Postfilter  (robust probe)  */
/* ------------------------------------------------------------------ */

void vtkParallelVectorsForBifurcationLine::Postfilter(
    vtkInformation *,
    vtkInformationVector **,
    vtkInformationVector *outputVector)
{
    printf("vtkParallelVectorsForBifurcationLine::Postfilter()\n");
    // Polyline output produced by vtkParallelVectors
    vtkPolyData *poly = vtkPolyData::SafeDownCast(
        outputVector->GetInformationObject(0)
            ->Get(vtkDataObject::DATA_OBJECT()));
    if (!poly || poly->GetNumberOfPoints() == 0)
        return;

    // Dataset that still carries the Jacobian (same input we were given)
    vtkDataSet *jacDs = vtkDataSet::SafeDownCast(this->GetInput());
    if (!jacDs || !this->Jacobian)
        return;

    //-------------------------------------------------------------------
    // 1) Probe Jacobian at every output point
    //-------------------------------------------------------------------
    vtkNew<vtkProbeFilter> probe;
    probe->SetSourceData(jacDs);
    probe->SetInputData(poly);
    probe->Update();
    vtkDataSet *probed = probe->GetOutput();
    vtkDataArray *Jarr = probed->GetPointData()->GetArray("Jacobian");
    if (!Jarr) // should never happen
        return;

    //-------------------------------------------------------------------
    // 2) Compute κ = –λ₁λ₂ for each point
    //-------------------------------------------------------------------
    vtkNew<vtkDoubleArray> kappa;
    kappa->SetName("bifurcation_criterion");
    kappa->SetNumberOfComponents(1);
    kappa->SetNumberOfTuples(Jarr->GetNumberOfTuples());

    for (vtkIdType pid = 0; pid < Jarr->GetNumberOfTuples(); ++pid)
    {
        double Jt[9];
        Jarr->GetTuple(pid, Jt);

        Eigen::Matrix<double, 3, 3> J;
        for (int k = 0; k < 9; ++k)
            J(k / 3, k % 3) = Jt[k];

        Eigen::EigenSolver<Eigen::Matrix<double, 3, 3>> es(J, false);
        std::array<double, 3> lam = {es.eigenvalues()[0].real(),
                                     es.eigenvalues()[1].real(),
                                     es.eigenvalues()[2].real()};
        std::sort(lam.begin(), lam.end(), std::greater<double>());
        kappa->SetValue(pid, -(lam[0] * lam[2]));
        printf("pid=%" PRId64 "  λ₁=%.3f  λ₂=%.3f  κ=%.3f\n",
               pid, lam[0], lam[1], kappa->GetValue(pid));
    }

    //-------------------------------------------------------------------
    // 3) Replace the operator’s placeholder array
    //-------------------------------------------------------------------
    poly->GetPointData()->RemoveArray("bifurcation_criterion");
    poly->GetPointData()->AddArray(kappa);
}

bool vtkParallelVectorsForVortexCore::AcceptSurfaceTriangle(
    const vtkIdType surfaceSimplexIndices[3])
{
    auto acceptedPoints = this->AcceptedPoints->GetPointer(0);
    return acceptedPoints[surfaceSimplexIndices[0]] && acceptedPoints[surfaceSimplexIndices[1]] &&
           acceptedPoints[surfaceSimplexIndices[2]];
}

/* ======================================================================= */
/*  Implementation – vtkBifurcationLine                                    */
/* ======================================================================= */

vtkStandardNewMacro(vtkBifurcationLine);

//-----------------------------------------------------------------------------
vtkBifurcationLine::vtkBifurcationLine()
    : PrimaryVectorFieldName(nullptr), SecondaryVectorFieldName(nullptr), EnableThreshold(false), MinimumCriterion(0.0), MaximumCriterion(VTK_DOUBLE_MAX)
{
    this->SetNumberOfInputPorts(1);
    this->SetNumberOfOutputPorts(1);
}

//-----------------------------------------------------------------------------
vtkBifurcationLine::~vtkBifurcationLine()
{
    this->SetPrimaryVectorFieldName(nullptr);
    this->SetSecondaryVectorFieldName(nullptr);
}

//-----------------------------------------------------------------------------
int vtkBifurcationLine::FillInputPortInformation(int, vtkInformation *info)
{
    info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkDataSet");
    return 1;
}

//-----------------------------------------------------------------------------
int vtkBifurcationLine::RequestData(
    vtkInformation *, vtkInformationVector **inVec, vtkInformationVector *outVec)
{
    /* -- 1. fetch input and primary field ------------------------------- */
    vtkDataSet *inDs = vtkDataSet::GetData(inVec[0]);

    if (!this->PrimaryVectorFieldName)
    {
        vtkErrorMacro("PrimaryVectorFieldName not set.");
        return 0;
    }
    vtkDataArray *v = inDs->GetPointData()->GetArray(this->PrimaryVectorFieldName);
    if (!v || v->GetNumberOfComponents() != 3)
    {
        vtkErrorMacro("Cannot find 3-component array \"" << this->PrimaryVectorFieldName << "\".");
        return 0;
    }

    /* -- 2. Jacobian ---------------------------------------------------- */
    vtkNew<vtkGradientFilter> grad;
    grad->SetInputData(inDs);
    grad->SetResultArrayName("Jacobian");
    grad->ComputeVorticityOff();
    grad->SetInputArrayToProcess(
        0, 0, 0, vtkDataObject::FIELD_ASSOCIATION_POINTS, v->GetName());
    grad->Update();
    vtkDataSet *jacDs = grad->GetOutput();
    vtkDataArray *jacob = jacDs->GetPointData()->GetArray("Jacobian");

    /* -- 3. secondary vector (acceleration) ----------------------------- */
    vtkDataArray *a = nullptr;
    if (this->SecondaryVectorFieldName &&
        jacDs->GetPointData()->HasArray(this->SecondaryVectorFieldName))
    {
        a = jacDs->GetPointData()->GetArray(this->SecondaryVectorFieldName);
    }
    else
    {
        vtkNew<vtkDoubleArray> acc;
        acc->SetName("Acceleration");
        acc->SetNumberOfComponents(3);
        acc->SetNumberOfTuples(v->GetNumberOfTuples());

        for (vtkIdType i = 0; i < v->GetNumberOfTuples(); ++i)
        {
            double vv[3], JJ[9], aa[3] = {0, 0, 0};
            v->GetTuple(i, vv);
            jacob->GetTuple(i, JJ);
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    aa[r] += JJ[3 * r + c] * vv[c];
            acc->SetTuple(i, aa);
        }
        jacDs->GetPointData()->AddArray(acc);
        a = acc;
    }

    /* -- 4. run parallel-vectors helper --------------------------------- */
    vtkNew<vtkParallelVectorsForBifurcationLine> pv;
    pv->SetInputData(jacDs);
    pv->SetJacobianDataArray(jacob);
    pv->SetFirstVectorFieldName(v->GetName());
    pv->SetSecondVectorFieldName(a->GetName());
    pv->SetEnableThreshold(this->EnableThreshold);
    pv->SetMinimumCriterion(this->MinimumCriterion);
    pv->SetMaximumCriterion(this->MaximumCriterion);
    pv->Update();

    /* -- 5. output ------------------------------------------------------ */
    vtkPolyData *outPd = vtkPolyData::GetData(outVec, 0);
    outPd->ShallowCopy(pv->GetOutput());
    return 1;
}

//-----------------------------------------------------------------------------
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
