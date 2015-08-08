//  Created by 乔磊 on 2015/8/7.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//

#ifndef __velocityPotential__LinearVelocityPotential__
#define __velocityPotential__LinearVelocityPotential__


#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/generic_linear_algebra.h>

//#define USE_PETSC_LA

namespace LA
{
#ifdef USE_PETSC_LA
  using namespace dealii::LinearAlgebraPETSc;
#else
  using namespace dealii::LinearAlgebraTrilinos;
#endif
}

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

#include <fstream>
#include <iostream>


namespace velocityPotential
{
  using namespace dealii;

  template <int dim>
  class LinearVelocityPotential
  {
  public:
    LinearVelocityPotential (
      const SmartPointer<parallel::distributed::Triangulation<dim> const > triangulation_in,
      const SmartPointer<NSFEMSolver::Parameters::AllParameters<dim> > parameters,
      const SmartPointer<LA::MPI::Vector> output_initial_field_ptr,
      MPI_Comm mpi_communicator_in);
    ~LinearVelocityPotential();

    void compute();
    void output_results() const;
    void transfer_solution (const FESystem<dim> &fe_NS,
                            const DoFHandler<dim> &dof_handler_NS,
                            LA::MPI::Vector  &NS_solution) const;
  private:
    void setup_system();
    void assemble_system();
    void solve();

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
    LA::MPI::Vector                           locally_relevant_solution;
    LA::MPI::Vector                           system_rhs;

    ConditionalOStream                        pcout;
    TimerOutput                               computing_timer;
    Point<3>                                  velocity_infty;

    class Postprocessor : public DataPostprocessor<dim>
    {
    public:
      Postprocessor (const Point<3> velocity_infty_in);

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
      Point<3>                                  velocity_infty;
    };


  };

}

#endif
