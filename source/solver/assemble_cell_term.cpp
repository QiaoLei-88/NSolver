//
//  NSolver::assemble_cell_term.cpp
//
//  Created by Lei Qiao on 15/8/9.
//  A work based on deal.II tutorial step-33.
//

#include <NSolver/solver/NSolver.h>

namespace NSFEMSolver
{
  using namespace dealii;


  // @sect4{NSolver::assemble_cell_term}
  //
  // This function assembles the cell term by computing the cell part of the
  // residual, adding its negative to the right hand side vector, and adding
  // its derivative with respect to the local variables to the Jacobian
  // (i.e. the Newton matrix). Recall that the cell contributions to the
  // residual read
  // $R_i = \left(\frac{\mathbf{w}^{k}_{n+1} - \mathbf{w}_n}{\delta t} ,
  // \mathbf{z}_i \right)_K $ $ +
  // \theta \mathbf{B}({\mathbf{w}^{k}_{n+1})(\mathbf{z}_i)_K $ $ +
  // (1-\theta) \mathbf{B}({\mathbf{w}_{n}) (\mathbf{z}_i)_K $ where
  // $\mathbf{B}({\mathbf{w})(\mathbf{z}_i)_K =
  // - \left(\mathbf{F}(\mathbf{w}),\nabla\mathbf{z}_i\right)_K $ $
  // + h^{\eta}(\nabla \mathbf{w} , \nabla \mathbf{z}_i)_K $ $
  // - (\mathbf{G}(\mathbf {w}), \mathbf{z}_i)_K $ for both
  // ${\mathbf{w} = \mathbf{w}^k_{n+1}$ and ${\mathbf{w} = \mathbf{w}_{n}}$ ,
  // $\mathbf{z}_i$ is the $i$th vector valued test function.
  //   Furthermore, the scalar product
  // $\left(\mathbf{F}(\mathbf{w}), \nabla\mathbf{z}_i\right)_K$ is
  // understood as $\int_K \sum_{c=1}^{\text{n\_components}}
  // \sum_{d=1}^{\text{dim}} \mathbf{F}(\mathbf{w})_{cd}
  // \frac{\partial z^c_i}{x_d}$ where $z^c_i$ is the $c$th component of
  // the $i$th test function.
  //
  //
  // At the top of this function, we do the usual housekeeping in terms of
  // allocating some local variables that we will need later. In particular,
  // we will allocate variables that will hold the values of the current
  // solution $W_{n+1}^k$ after the $k$th Newton iteration (variable
  // <code>W</code>) and the previous time step's solution $W_{n}$ (variable
  // <code>W_old</code>).
  //
  // In addition to these, we need the gradients of the current variables.  It
  // is a bit of a shame that we have to compute these; we almost don't.  The
  // nice thing about a simple conservation law is that the flux doesn't
  // generally involve any gradients.  We do need these, however, for the
  // diffusion stabilization.
  //
  // The actual format in which we store these variables requires some
  // explanation. First, we need values at each quadrature point for each of
  // the <code>EulerEquations::n_components</code> components of the solution
  // vector. This makes for a two-dimensional table for which we use deal.II's
  // Table class (this is more efficient than
  // <code>std::vector@<std::vector@<T@> @></code> because it only needs to
  // allocate memory once, rather than once for each element of the outer
  // vector). Similarly, the gradient is a three-dimensional table, which the
  // Table class also supports.
  //
  // Secondly, we want to use automatic differentiation. To this end, we use
  // the Sacado::Fad::DFad template for everything that is computed from the
  // variables with respect to which we would like to compute
  // derivatives. This includes the current solution and gradient at the
  // quadrature points (which are linear combinations of the degrees of
  // freedom) as well as everything that is computed from them such as the
  // residual, but not the previous time step's solution. These variables are
  // all found in the first part of the function, along with a variable that
  // we will use to store the derivatives of a single component of the
  // residual:
  template <int dim>
  void
  NSolver<dim>::
  assemble_cell_term (const FEValues<dim>             &fe_v,
                      const std::vector<types::global_dof_index> &dof_indices)
  {
    const unsigned int dofs_per_cell = fe_v.dofs_per_cell;
    const unsigned int n_q_points    = fe_v.n_quadrature_points;

    Table<2,Sacado::Fad::DFad<double> >
    W (n_q_points, EquationComponents<dim>::n_components);

    Table<2,double>
    W_old (n_q_points, EquationComponents<dim>::n_components);

    Table<3,Sacado::Fad::DFad<double> >
    grad_W (n_q_points, EquationComponents<dim>::n_components, dim);
    Table<3,double>
    grad_W_old (n_q_points, EquationComponents<dim>::n_components, dim);

    // Next, we have to define the independent variables that we will try to
    // determine by solving a Newton step. These independent variables are the
    // values of the local degrees of freedom which we extract here:
    std::vector<Sacado::Fad::DFad<double> > independent_local_dof_values (dofs_per_cell);
    for (unsigned int i=0; i<dofs_per_cell; ++i)
      {
        independent_local_dof_values[i] = current_solution (dof_indices[i]);
      }

    // The next step incorporates all the magic: we declare a subset of the
    // autodifferentiation variables as independent degrees of freedom,
    // whereas all the other ones remain dependent functions. These are
    // precisely the local degrees of freedom just extracted. All calculations
    // that reference them (either directly or indirectly) will accumulate
    // sensitivities with respect to these variables.
    //
    // In order to mark the variables as independent, the following does the
    // trick, marking <code>independent_local_dof_values[i]</code> as the
    // $i$th independent variable out of a total of
    // <code>dofs_per_cell</code>:
    for (unsigned int i=0; i<dofs_per_cell; ++i)
      {
        independent_local_dof_values[i].diff (i, dofs_per_cell);
      }

    // After all these declarations, let us actually compute something. First,
    // the values of <code>W</code>, <code>W_old</code>, <code>grad_W</code>
    // and <code>grad_W_old</code>, which we can compute from the local DoF values
    // by using the formula $W(x_q)=\sum_i \mathbf W_i \Phi_i(x_q)$, where
    // $\mathbf W_i$ is the $i$th entry of the (local part of the) solution
    // vector, and $\Phi_i(x_q)$ the value of the $i$th vector-valued shape
    // function evaluated at quadrature point $x_q$. The gradient can be
    // computed in a similar way.
    //
    // Ideally, we could compute this information using a call into something
    // like FEValues::get_function_values and FEValues::get_function_gradients,
    // but since (i) we would have to extend the FEValues class for this, and
    // (ii) we don't want to make the entire <code>old_solution</code> vector
    // fad types, only the local cell variables, we explicitly code the loop
    // above. Before this, we add another loop that initializes all the fad
    // variables to zero:
    for (unsigned int q=0; q<n_q_points; ++q)
      for (unsigned int c=0; c<EquationComponents<dim>::n_components; ++c)
        {
          W[q][c]       = 0;
          W_old[q][c]   = 0;
          for (unsigned int d=0; d<dim; ++d)
            {
              grad_W[q][c][d] = 0;
              grad_W_old[q][c][d] = 0;
            }
        }

    for (unsigned int q=0; q<n_q_points; ++q)
      for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
          const unsigned int c = fe_v.get_fe().system_to_component_index (i).first;

          W[q][c] += independent_local_dof_values[i] *
                     fe_v.shape_value_component (i, q, c);
          W_old[q][c] += old_solution (dof_indices[i]) *
                         fe_v.shape_value_component (i, q, c);

          for (unsigned int d = 0; d < dim; d++)
            {
              grad_W[q][c][d] += independent_local_dof_values[i] *
                                 fe_v.shape_grad_component (i, q, c)[d];
              grad_W_old[q][c][d] += old_solution (dof_indices[i]) *
                                     fe_v.shape_grad_component (i, q, c)[d];
            }
        }


    // Next, in order to compute the cell contributions, we need to evaluate
    // $F({\mathbf w}^k_{n+1})$, $G({\mathbf w}^k_{n+1})$ and
    // $F({\mathbf w}_n)$, $G({\mathbf w}_n)$ at all quadrature
    // points. To store these, we also need to allocate a bit of memory. Note
    // that we compute the flux matrices and right hand sides in terms of
    // autodifferentiation variables, so that the Jacobian contributions can
    // later easily be computed from it:
    std::vector <
    std_cxx11::array <std_cxx11::array <Sacado::Fad::DFad<double>, dim>, EquationComponents<dim>::n_components >
    > flux (n_q_points);

    std::vector <
    std_cxx11::array <std_cxx11::array <Sacado::Fad::DFad<double>, dim>, EquationComponents<dim>::n_components >
    > visc_flux (n_q_points);

    std::vector <
    std_cxx11::array <std_cxx11::array <double, dim>, EquationComponents<dim>::n_components >
    > flux_old (n_q_points);

    std::vector < std_cxx11::array< Sacado::Fad::DFad<double>, EquationComponents<dim>::n_components> > forcing (
      n_q_points);

    std::vector < std_cxx11::array< double, EquationComponents<dim>::n_components> > forcing_old (n_q_points);

    //MMS: evaluate source term
    std::vector <typename MMS<dim>::F_V> mms_source (n_q_points);
    std::vector <typename MMS<dim>::F_V> mms_value (n_q_points);
    std::vector <typename MMS<dim>::F_T> mms_grad (n_q_points);
    if (parameters->n_mms == 1)
      {
        for (unsigned int q=0; q<n_q_points; ++q)
          {
            mms.evaluate (fe_v.quadrature_point (q), mms_value[q], mms_grad[q], mms_source[q], /* const bool need_source = */ true);
          }
      }
    // TODO:
    // viscosity_old needed here.
    // Vector for viscosity should be sized to n_dof rather than n_active_cell, only entropy viscosity is
    // cellwise, while physical viscosity is per-dof.

    const double mu =
      artificial_viscosity[fe_v.get_cell()->active_cell_index()];
    const double prandtlNumber = 0.72;
    const double kappa = mu / (prandtlNumber * (parameters->gas_gamma - 1.0));

    for (unsigned int q=0; q<n_q_points; ++q)
      {
        EulerEquations<dim>::compute_inviscid_flux (W_old[q], flux_old[q]);
        EulerEquations<dim>::compute_forcing_vector (W_old[q], forcing_old[q]);
        EulerEquations<dim>::compute_inviscid_flux (W[q], flux[q]);
        EulerEquations<dim>::compute_viscous_flux (W[q], grad_W[q], visc_flux[q], mu, kappa);
        EulerEquations<dim>::compute_forcing_vector (W[q], forcing[q]);
      }

    // Avoid waring on unused parameter
    (void)nonlin_iter;

    // We now have all of the pieces in place, so perform the assembly.  We
    // have an outer loop through the components of the system, and an inner
    // loop over the quadrature points, where we accumulate contributions to
    // the $i$th residual $R_i$. The general formula for this residual is
    // given in the introduction and at the top of this function. We can,
    // however, simplify it a bit taking into account that the $i$th
    // (vector-valued) test function $\mathbf{z}_i$ has in reality only a
    // single nonzero component (more on this topic can be found in the @ref
    // vector_valued module). It will be represented by the variable
    // <code>component_i</code> below. With this, the residual term can be
    // re-written as
    // @f{eqnarray*}
    // R_i &=&
    // \left(\frac{(\mathbf{w}_{n+1} -
    // \mathbf{w}_n)_{\text{component\_i}}}{\delta
    // t},(\mathbf{z}_i)_{\text{component\_i}}\right)_K
    // \\ &-& \sum_{d=1}^{\text{dim}} \left(  \theta \mathbf{F}
    // ({\mathbf{w^k_{n+1}}})_{\text{component\_i},d} + (1-\theta)
    // \mathbf{F} ({\mathbf{w_{n}}})_{\text{component\_i},d}  ,
    // \frac{\partial(\mathbf{z}_i)_{\text{component\_i}}} {\partial
    // x_d}\right)_K
    // \\ &+& \sum_{d=1}^{\text{dim}} h^{\eta} \left( \theta \frac{\partial
    // \mathbf{w^k_{n+1}}_{\text{component\_i}}}{\partial x_d} + (1-\theta)
    // \frac{\partial \mathbf{w_n}_{\text{component\_i}}}{\partial x_d} ,
    // \frac{\partial (\mathbf{z}_i)_{\text{component\_i}}}{\partial x_d} \right)_K
    // \\ &-& \left( \theta\mathbf{G}({\mathbf{w}^k_n+1} )_{\text{component\_i}} +
    // (1-\theta)\mathbf{G}({\mathbf{w}_n} )_{\text{component\_i}} ,
    // (\mathbf{z}_i)_{\text{component\_i}} \right)_K ,
    // @f}
    // where integrals are
    // understood to be evaluated through summation over quadrature points.
    //
    // We initially sum all contributions of the residual in the positive
    // sense, so that we don't need to negative the Jacobian entries.  Then,
    // when we sum into the <code>right_hand_side</code> vector, we negate
    // this residual.
    const double dt = parameters->use_local_time_step_size ?
                      local_time_step_size[fe_v.get_cell()->active_cell_index()]
                      :
                      global_time_step_size;
    for (unsigned int i=0; i<fe_v.dofs_per_cell; ++i)
      {
        Sacado::Fad::DFad<double> R_i = 0;
        double cell_physical_residual = 0.0;

        const unsigned int
        component_i = fe_v.get_fe().system_to_component_index (i).first;

        // The residual for each row (i) will be accumulating into this fad
        // variable.  At the end of the assembly for this row, we will query
        // for the sensitivities to this variable and add them into the
        // Jacobian.

        for (unsigned int point=0; point<fe_v.n_quadrature_points; ++point)
          {
            std_cxx11::array<Sacado::Fad::DFad<double> , EquationComponents<dim>::n_components> w_conservative;
            std_cxx11::array<double, EquationComponents<dim>::n_components> w_conservative_old;
            EulerEquations<dim>::compute_conservative_vector (W[point], w_conservative);
            EulerEquations<dim>::compute_conservative_vector (W_old[point], w_conservative_old);

            // TODO: accumulate R_i first and the multiply with shape_value_component * JxW together.
            if (!parameters->turn_off_time_marching)
              {
                const Sacado::Fad::DFad<double> tmp =
                  1.0 / dt *
                  (w_conservative[component_i] - w_conservative_old[component_i]) *
                  fe_v.shape_value_component (i, point, component_i) *
                  fe_v.JxW (point);
                R_i += tmp;
                if (!parameters->is_steady)
                  {
                    cell_physical_residual += tmp.val();
                  }
              }

            for (unsigned int d=0; d<dim; d++)
              {
                {
                  const Sacado::Fad::DFad<double> tmp =
                    (parameters->theta * flux[point][component_i][d] +
                     (1.0-parameters->theta) * flux_old[point][component_i][d]) *
                    fe_v.shape_grad_component (i, point, component_i)[d] *
                    fe_v.JxW (point);
                  R_i -= tmp;
                  cell_physical_residual -= tmp.val();
                }

                R_i += visc_flux[point][component_i][d] *
                       fe_v.shape_grad_component (i, point, component_i)[d] *
                       fe_v.JxW (point);
              }
            {
              const Sacado::Fad::DFad<double> tmp =
                (parameters->theta  * forcing[point][component_i] +
                 (1.0 - parameters->theta) * forcing_old[point][component_i]) *
                fe_v.shape_value_component (i, point, component_i) *
                fe_v.JxW (point);
              R_i -= tmp;
              cell_physical_residual -= tmp.val();
            }
            if (parameters->n_mms == 1)
              {
                //MMS: apply MMS source term
                const Sacado::Fad::DFad<double> tmp =
                  mms_source[point][component_i] *
                  fe_v.shape_value_component (i, point, component_i) *
                  fe_v.JxW (point);
                R_i -= tmp;
                cell_physical_residual -= tmp.val();
              }
            if (parameters->laplacian_continuation > 0.0 &&
                laplacian_coefficient > 0.0)
              {
                for (unsigned int d=0; d<dim; d++)
                  {
                    R_i += laplacian_coefficient *
                           grad_W[point][component_i][d] *
                           fe_v.shape_grad_component (i, point, component_i)[d];
                  }
              }
          }

        // At the end of the loop, we add the sensitivities to the matrix and
        // subtract the residual from the right hand side.
        // Trilinos FAD data type stores all the derivatives in an array.
        // We can pass them to the system matrix directly.
        system_matrix.add (dof_indices[i], dof_indices.size(),
                           & (dof_indices[0]), & (R_i.fastAccessDx (0)));
        right_hand_side (dof_indices[i]) -= R_i.val();
        physical_residual (dof_indices[i]) -= cell_physical_residual;
      }
  }

#include "NSolver.inst"
}
