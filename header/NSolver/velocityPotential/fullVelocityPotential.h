//  Created by ÇÇÀÚ on 2015/9/8.
//  Copyright (c) 2015Äê ÇÇÀÚ. All rights reserved.
//

#ifndef __velocityPotential__FullVelocityPotential__
#define __velocityPotential__FullVelocityPotential__


#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/generic_linear_algebra.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/lac/petsc_parallel_sparse_matrix.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/base/smartpointer.h>

#include <NSolver/Parameters/AllParameters.h>
#include <NSolver/EquationComponents.h>
#include <NSolver/types.h>

#include <fstream>
#include <iostream>

namespace velocityPotential
{
  using namespace dealii;

  template <int dim>
  class FullVelocityPotential
  {
  public:
    FullVelocityPotential (
      const SmartPointer<parallel::distributed::Triangulation<dim> const > triangulation_in,
      const SmartPointer<NSFEMSolver::Parameters::AllParameters<dim> const> parameters,
      MPI_Comm mpi_communicator_in);
    ~FullVelocityPotential();

    void compute();
    void output_results() const;
    void transfer_solution (const FESystem<dim> &fe_NS,
                            const DoFHandler<dim> &dof_handler_NS,
                            LA::MPI::Vector  &NS_solution) const;
  private:
    void setup_system();
    void assemble_system();
    void solve (double &final_residual);

    MPI_Comm                                  mpi_communicator;
    const SmartPointer<NSFEMSolver::Parameters::AllParameters<dim> const>    parameters;

    SmartPointer <
    parallel::distributed::Triangulation<dim> const> const triangulation;

    DoFHandler<dim>                           dof_handler;
    FE_Q<dim>                                 fe;

    IndexSet                                  locally_owned_dofs;
    IndexSet                                  locally_relevant_dofs;

    ConstraintMatrix                          constraints;

    LA::MPI::SparseMatrix                     system_matrix;
    LA::MPI::Vector                           locally_owned_solution;
    LA::MPI::Vector                           newton_update;
    LA::MPI::Vector                           system_rhs;
    LA::MPI::Vector                           locally_relevant_solution;

    ConditionalOStream                        pcout;
    TimerOutput                               computing_timer;
    Tensor<1,dim>                             velocity_infty;
    const double                              Mach_infty;
    const double                              Mach_infty_square;
    const double                              gas_gamma;
    const double                              gm1;
    const double                              const_a;

    class Postprocessor : public DataPostprocessor<dim>
    {
    public:
      Postprocessor (const double Mach_infty_square_in,
                     const double gas_gamma_in);

      virtual void compute_derived_quantities_scalar (const std::vector<double>             &uh,
                                                      const std::vector<Tensor<1,dim> >     &duh,
                                                      const std::vector<Tensor<2,dim> >     &dduh,
                                                      const std::vector<Point<dim> >        &normals,
                                                      const std::vector<Point<dim> >        &points,
                                                      std::vector<Vector<double> >          &computed_quantities) const;

      virtual std::vector<std::string> get_names() const;

      virtual std::vector<DataComponentInterpretation::DataComponentInterpretation>
      get_data_component_interpretation() const;

      virtual UpdateFlags get_needed_update_flags() const;
    private:
      const double Mach_infty_square;
      const double gas_gamma;
      const double gm1;
    };
  };

  namespace internal
  {
    template<int dim>
    inline
    double compute_density (const Tensor<1,dim> &v,
                            const double gm1,
                            const double Mach_infty_square)
    {
      return (std::pow (1.0-0.5*gm1* (v.norm_square() - Mach_infty_square), 1.0/gm1));
    }
  }
}

#endif
