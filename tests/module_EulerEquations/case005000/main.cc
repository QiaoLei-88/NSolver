#include <NSolver/NSEquation.h>
#include <fstream>

#define dim 2

using namespace dealii;
using namespace NSFEMSolver;

int main (int argc, char *argv[])
{
  const unsigned int dimension (2);

  char *input_file (0);
  char default_input_file[10] = "input.prm";
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

    solver_parameters.declare_parameters (prm);
    prm.read_input (input_file);
    solver_parameters.parse_parameters (prm);
  }
  EulerEquations<dim>::gas_gamma = 1.4;
  EulerEquations<dim>::set_parameter (&solver_parameters);

  Tensor<1,dim>       normal_vector;
  Table<2,double>     Wplus (1,EquationComponents<dim>::n_components);
  Vector<double>      boundary_values (EquationComponents<dim>::n_components);
  Table<2,double>     Wminus (1,EquationComponents<dim>::n_components);

  std::ofstream fout ("all.out");

  {
    normal_vector[0] = 0.0;
    normal_vector[1] = 1.0;

    Wplus[0][0] = 0.29999;
    Wplus[0][1] = -1.00746e-06;
    Wplus[0][2] = 1.0;
    Wplus[0][3] = 0.714286;

    fout << "Face norm: [" << normal_vector[0] << ", " << normal_vector[1] << "]\n";
    fout << "Wplus:\n";
    for (unsigned int ic=0; ic<EquationComponents<dim>::n_components; ++ic)
      {
        fout << Wplus[0][ic] << ", ";
      }
    fout << std::endl;
    EulerEquations<dim>::compute_Wminus (NSFEMSolver::Boundary::FarField,
                                         normal_vector,
                                         Wplus[0],
                                         boundary_values,
                                         Wminus[0]);
    fout << "Wminus:\n";
    for (unsigned int ic=0; ic<EquationComponents<dim>::n_components; ++ic)
      {
        fout << Wminus[0][ic] << ", ";
      }
    fout << std::endl;
    fout << std::endl;
  }

  {
    normal_vector[0] = 1.0;
    normal_vector[1] = 0.0;

    Wplus[0][0] = 0.29999;
    Wplus[0][1] = -1.00746e-06;
    Wplus[0][2] = 1.0;
    Wplus[0][3] = 0.714286;

    fout << "Face norm: [" << normal_vector[0] << ", " << normal_vector[1] << "]\n";
    fout << "Wplus:\n";
    for (unsigned int ic=0; ic<EquationComponents<dim>::n_components; ++ic)
      {
        fout << Wplus[0][ic] << ", ";
      }
    fout << std::endl;
    EulerEquations<dim>::compute_Wminus (NSFEMSolver::Boundary::FarField,
                                         normal_vector,
                                         Wplus[0],
                                         boundary_values,
                                         Wminus[0]);
    fout << "Wminus:\n";
    for (unsigned int ic=0; ic<EquationComponents<dim>::n_components; ++ic)
      {
        fout << Wminus[0][ic] << ", ";
      }
    fout << std::endl;
    fout << std::endl;
  }

  {
    normal_vector[0] = 0.0;
    normal_vector[1] = 1.0;

    Wplus[0][0] = 0.3;
    Wplus[0][1] = -1.00746e-06;
    Wplus[0][2] = 1.0;
    Wplus[0][3] = 0.714286;

    fout << "Face norm: [" << normal_vector[0] << ", " << normal_vector[1] << "]\n";
    fout << "Wplus:\n";
    for (unsigned int ic=0; ic<EquationComponents<dim>::n_components; ++ic)
      {
        fout << Wplus[0][ic] << ", ";
      }
    fout << std::endl;
    EulerEquations<dim>::compute_Wminus (NSFEMSolver::Boundary::FarField,
                                         normal_vector,
                                         Wplus[0],
                                         boundary_values,
                                         Wminus[0]);
    fout << "Wminus:\n";
    for (unsigned int ic=0; ic<EquationComponents<dim>::n_components; ++ic)
      {
        fout << Wminus[0][ic] << ", ";
      }
    fout << std::endl;
    fout << std::endl;
  }

  {
    normal_vector[0] = 0.0;
    normal_vector[1] = 1.0;

    Wplus[0][0] = 0.29999;
    Wplus[0][1] = 0.0;
    Wplus[0][2] = 1.0;
    Wplus[0][3] = 0.714286;

    fout << "Face norm: [" << normal_vector[0] << ", " << normal_vector[1] << "]\n";
    fout << "Wplus:\n";
    for (unsigned int ic=0; ic<EquationComponents<dim>::n_components; ++ic)
      {
        fout << Wplus[0][ic] << ", ";
      }
    fout << std::endl;
    EulerEquations<dim>::compute_Wminus (NSFEMSolver::Boundary::FarField,
                                         normal_vector,
                                         Wplus[0],
                                         boundary_values,
                                         Wminus[0]);
    fout << "Wminus:\n";
    for (unsigned int ic=0; ic<EquationComponents<dim>::n_components; ++ic)
      {
        fout << Wminus[0][ic] << ", ";
      }
    fout << std::endl;
    fout << std::endl;
  }

  {
    normal_vector[0] = -1.0;
    normal_vector[1] = 0.0;

    Wplus[0][0] = 0.7;
    Wplus[0][1] = -1.00746e-06;
    Wplus[0][2] = 1.0;
    Wplus[0][3] = 0.714286;

    fout << "Face norm: [" << normal_vector[0] << ", " << normal_vector[1] << "]\n";
    fout << "Wplus:\n";
    for (unsigned int ic=0; ic<EquationComponents<dim>::n_components; ++ic)
      {
        fout << Wplus[0][ic] << ", ";
      }
    fout << std::endl;
    EulerEquations<dim>::compute_Wminus (NSFEMSolver::Boundary::FarField,
                                         normal_vector,
                                         Wplus[0],
                                         boundary_values,
                                         Wminus[0]);
    fout << "Wminus:\n";
    for (unsigned int ic=0; ic<EquationComponents<dim>::n_components; ++ic)
      {
        fout << Wminus[0][ic] << ", ";
      }
    fout << std::endl;
    fout << std::endl;
  }

  {
    normal_vector[0] = 0.0;
    normal_vector[1] = -1.0;

    Wplus[0][0] = 0.29999;
    Wplus[0][1] = -1.00746e-06;
    Wplus[0][2] = 1.0;
    Wplus[0][3] = 0.714286;

    fout << "Face norm: [" << normal_vector[0] << ", " << normal_vector[1] << "]\n";
    fout << "Wplus:\n";
    for (unsigned int ic=0; ic<EquationComponents<dim>::n_components; ++ic)
      {
        fout << Wplus[0][ic] << ", ";
      }
    fout << std::endl;
    EulerEquations<dim>::compute_Wminus (NSFEMSolver::Boundary::FarField,
                                         normal_vector,
                                         Wplus[0],
                                         boundary_values,
                                         Wminus[0]);
    fout << "Wminus:\n";
    for (unsigned int ic=0; ic<EquationComponents<dim>::n_components; ++ic)
      {
        fout << Wminus[0][ic] << ", ";
      }
    fout << std::endl;
    fout << std::endl;
  }

  {
    normal_vector[0] = -1.0;
    normal_vector[1] = 0.0;

    Wplus[0][0] = 0.29999;
    Wplus[0][1] = -1.00746e-06;
    Wplus[0][2] = 1.0;
    Wplus[0][3] = 0.714286;

    fout << "Face norm: [" << normal_vector[0] << ", " << normal_vector[1] << "]\n";
    fout << "Wplus:\n";
    for (unsigned int ic=0; ic<EquationComponents<dim>::n_components; ++ic)
      {
        fout << Wplus[0][ic] << ", ";
      }
    fout << std::endl;
    EulerEquations<dim>::compute_Wminus (NSFEMSolver::Boundary::FarField,
                                         normal_vector,
                                         Wplus[0],
                                         boundary_values,
                                         Wminus[0]);
    fout << "Wminus:\n";
    for (unsigned int ic=0; ic<EquationComponents<dim>::n_components; ++ic)
      {
        fout << Wminus[0][ic] << ", ";
      }
    fout << std::endl;
    fout << std::endl;
  }

  fout.close();
  return (0);
}
