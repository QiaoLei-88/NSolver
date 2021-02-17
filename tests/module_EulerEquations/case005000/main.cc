#include <NSolver/NSEquation.h>

#include <fstream>

#define dim 2

using namespace dealii;
using namespace NSFEMSolver;

int
main(int argc, char *argv[])
{
  const unsigned int dimension(2);

  char *input_file(0);
  char  default_input_file[10] = "input.prm";
  if (argc < 2)
    {
      input_file = default_input_file;
    }
  else
    {
      input_file = argv[1];
    }

  Parameters::AllParameters<dimension> solver_parameters;
  {
    ParameterHandler prm;

    solver_parameters.declare_parameters(prm);
    prm.parse_input(input_file);
    solver_parameters.parse_parameters(prm);
  }
  EulerEquations<dim>::set_parameter(&solver_parameters);

  Tensor<1, dim> normal_vector;
  Vector<double> Wplus(EquationComponents<dim>::n_components);
  Vector<double> boundary_values(EquationComponents<dim>::n_components);
  Vector<double> Wminus(EquationComponents<dim>::n_components);

  std::ofstream fout("output.out");

  {
    normal_vector[0] = 0.0;
    normal_vector[1] = 1.0;

    Wplus[0] = 0.29999;
    Wplus[1] = -1.00746e-06;
    Wplus[2] = 1.0;
    Wplus[3] = 0.714286;

    fout << "Face norm: [" << normal_vector[0] << ", " << normal_vector[1]
         << "]\n";
    fout << "Wplus:\n";
    for (unsigned int ic = 0; ic < EquationComponents<dim>::n_components; ++ic)
      {
        fout << Wplus[ic] << ", ";
      }
    fout << std::endl;
    EulerEquations<dim>::compute_Wminus(NSFEMSolver::Boundary::FarField,
                                        normal_vector,
                                        Wplus,
                                        boundary_values,
                                        Wminus);
    fout << "Wminus:\n";
    for (unsigned int ic = 0; ic < EquationComponents<dim>::n_components; ++ic)
      {
        fout << Wminus[ic] << ", ";
      }
    fout << std::endl;
    fout << std::endl;
  }

  {
    normal_vector[0] = 1.0;
    normal_vector[1] = 0.0;

    Wplus[0] = 0.29999;
    Wplus[1] = -1.00746e-06;
    Wplus[2] = 1.0;
    Wplus[3] = 0.714286;

    fout << "Face norm: [" << normal_vector[0] << ", " << normal_vector[1]
         << "]\n";
    fout << "Wplus:\n";
    for (unsigned int ic = 0; ic < EquationComponents<dim>::n_components; ++ic)
      {
        fout << Wplus[ic] << ", ";
      }
    fout << std::endl;
    EulerEquations<dim>::compute_Wminus(NSFEMSolver::Boundary::FarField,
                                        normal_vector,
                                        Wplus,
                                        boundary_values,
                                        Wminus);
    fout << "Wminus:\n";
    for (unsigned int ic = 0; ic < EquationComponents<dim>::n_components; ++ic)
      {
        fout << Wminus[ic] << ", ";
      }
    fout << std::endl;
    fout << std::endl;
  }

  {
    normal_vector[0] = 0.0;
    normal_vector[1] = 1.0;

    Wplus[0] = 0.3;
    Wplus[1] = -1.00746e-06;
    Wplus[2] = 1.0;
    Wplus[3] = 0.714286;

    fout << "Face norm: [" << normal_vector[0] << ", " << normal_vector[1]
         << "]\n";
    fout << "Wplus:\n";
    for (unsigned int ic = 0; ic < EquationComponents<dim>::n_components; ++ic)
      {
        fout << Wplus[ic] << ", ";
      }
    fout << std::endl;
    EulerEquations<dim>::compute_Wminus(NSFEMSolver::Boundary::FarField,
                                        normal_vector,
                                        Wplus,
                                        boundary_values,
                                        Wminus);
    fout << "Wminus:\n";
    for (unsigned int ic = 0; ic < EquationComponents<dim>::n_components; ++ic)
      {
        fout << Wminus[ic] << ", ";
      }
    fout << std::endl;
    fout << std::endl;
  }

  {
    normal_vector[0] = 0.0;
    normal_vector[1] = 1.0;

    Wplus[0] = 0.29999;
    Wplus[1] = 0.0;
    Wplus[2] = 1.0;
    Wplus[3] = 0.714286;

    fout << "Face norm: [" << normal_vector[0] << ", " << normal_vector[1]
         << "]\n";
    fout << "Wplus:\n";
    for (unsigned int ic = 0; ic < EquationComponents<dim>::n_components; ++ic)
      {
        fout << Wplus[ic] << ", ";
      }
    fout << std::endl;
    EulerEquations<dim>::compute_Wminus(NSFEMSolver::Boundary::FarField,
                                        normal_vector,
                                        Wplus,
                                        boundary_values,
                                        Wminus);
    fout << "Wminus:\n";
    for (unsigned int ic = 0; ic < EquationComponents<dim>::n_components; ++ic)
      {
        fout << Wminus[ic] << ", ";
      }
    fout << std::endl;
    fout << std::endl;
  }

  {
    normal_vector[0] = -1.0;
    normal_vector[1] = 0.0;

    Wplus[0] = 0.7;
    Wplus[1] = -1.00746e-06;
    Wplus[2] = 1.0;
    Wplus[3] = 0.714286;

    fout << "Face norm: [" << normal_vector[0] << ", " << normal_vector[1]
         << "]\n";
    fout << "Wplus:\n";
    for (unsigned int ic = 0; ic < EquationComponents<dim>::n_components; ++ic)
      {
        fout << Wplus[ic] << ", ";
      }
    fout << std::endl;
    EulerEquations<dim>::compute_Wminus(NSFEMSolver::Boundary::FarField,
                                        normal_vector,
                                        Wplus,
                                        boundary_values,
                                        Wminus);
    fout << "Wminus:\n";
    for (unsigned int ic = 0; ic < EquationComponents<dim>::n_components; ++ic)
      {
        fout << Wminus[ic] << ", ";
      }
    fout << std::endl;
    fout << std::endl;
  }

  {
    normal_vector[0] = 0.0;
    normal_vector[1] = -1.0;

    Wplus[0] = 0.29999;
    Wplus[1] = -1.00746e-06;
    Wplus[2] = 1.0;
    Wplus[3] = 0.714286;

    fout << "Face norm: [" << normal_vector[0] << ", " << normal_vector[1]
         << "]\n";
    fout << "Wplus:\n";
    for (unsigned int ic = 0; ic < EquationComponents<dim>::n_components; ++ic)
      {
        fout << Wplus[ic] << ", ";
      }
    fout << std::endl;
    EulerEquations<dim>::compute_Wminus(NSFEMSolver::Boundary::FarField,
                                        normal_vector,
                                        Wplus,
                                        boundary_values,
                                        Wminus);
    fout << "Wminus:\n";
    for (unsigned int ic = 0; ic < EquationComponents<dim>::n_components; ++ic)
      {
        fout << Wminus[ic] << ", ";
      }
    fout << std::endl;
    fout << std::endl;
  }

  {
    normal_vector[0] = -1.0;
    normal_vector[1] = 0.0;

    Wplus[0] = 0.29999;
    Wplus[1] = -1.00746e-06;
    Wplus[2] = 1.0;
    Wplus[3] = 0.714286;

    fout << "Face norm: [" << normal_vector[0] << ", " << normal_vector[1]
         << "]\n";
    fout << "Wplus:\n";
    for (unsigned int ic = 0; ic < EquationComponents<dim>::n_components; ++ic)
      {
        fout << Wplus[ic] << ", ";
      }
    fout << std::endl;
    EulerEquations<dim>::compute_Wminus(NSFEMSolver::Boundary::FarField,
                                        normal_vector,
                                        Wplus,
                                        boundary_values,
                                        Wminus);
    fout << "Wminus:\n";
    for (unsigned int ic = 0; ic < EquationComponents<dim>::n_components; ++ic)
      {
        fout << Wminus[ic] << ", ";
      }
    fout << std::endl;
    fout << std::endl;
  }

  fout.close();
  return (0);
}
