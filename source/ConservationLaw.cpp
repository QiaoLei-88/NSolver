//
//  prj01-Newton01.cpp
//  prj01-Newton2D
//
//  Created by Lei Qiao on 15/2/3.
//  A work based on deal.II turorial step-33.
//

/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2007 - 2013 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Author: David Neckels, Boulder, Colorado, 2007, 2008
 */


#include "ConservationLaw.h"
#include "NSEquation.h"

namespace Step33
{
  using namespace dealii;

  // @sect3{Conservation law class}

  // Here finally comes the class that actually does something with all the
  // Euler equation and parameter specifics we've defined above. The public
  // interface is pretty much the same as always (the constructor now takes
  // the name of a file from which to read parameters, which is passed on the
  // command line). The private function interface is also pretty similar to
  // the usual arrangement, with the <code>assemble_system</code> function
  // split into three parts: one that contains the main loop over all cells
  // and that then calls the other two for integrals over cells and faces,
  // respectively.


  // @sect4{ConservationLaw::ConservationLaw}
  //
  // There is nothing much to say about the constructor. Essentially, it reads
  // the input file and fills the parameter object with the parsed values:
  template <int dim>
  ConservationLaw<dim>::ConservationLaw (const char *input_filename)
  :
  mapping (),
  fe (FE_Q<dim>(1), EulerEquations<dim>::n_components),
  dof_handler (triangulation),
  quadrature (2),
  face_quadrature (2),
  verbose_cout (std::cout, false)
  {
  ParameterHandler prm;
  Parameters::AllParameters<dim>::declare_parameters (prm);

  prm.read_input (input_filename);
  parameters.parse_parameters (prm);
  parameters.time_step_factor = 1.0;

  verbose_cout.set_condition (parameters.output == Parameters::Solver::verbose);
  }



  // @sect4{ConservationLaw::setup_system}
  //
  // The following (easy) function is called each time the mesh is
  // changed. All it does is to resize the Trilinos matrix according to a
  // sparsity pattern that we generate as in all the previous tutorial
  // programs.
  template <int dim>
  void ConservationLaw<dim>::setup_system ()
  {
  CompressedSparsityPattern sparsity_pattern (dof_handler.n_dofs(),
                                              dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern (dof_handler, sparsity_pattern);

  system_matrix.reinit (sparsity_pattern);
  }

  // @sect4{ConservationLaw::calc_time_step}
  //
  // Determian time step size of next time step.
  template <int dim>
  void ConservationLaw<dim>::calc_time_step ()
  {
  if (parameters.is_rigid_timestep_size)
    {
    parameters.time_step = parameters.readin_time_step;
    }
  else
    {
    FEValues<dim> fe_v (mapping, fe, quadrature, update_values);
    const unsigned int   n_q_points = fe_v.n_quadrature_points;
    std::vector<Vector<double> > solution_values(n_q_points,
                                                 Vector<double>(dim+2));
    double min_time_step = parameters.readin_time_step;
    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (; cell!=endc; ++cell)
      {
      fe_v.reinit (cell);
      fe_v.get_function_values (current_solution, solution_values);
      const double cell_size = fe_v.get_cell()->diameter();
      double velocity;
      for (unsigned int q=0; q<n_q_points; ++q)
        {
        const double density = solution_values[q](EulerEquations<dim>::density_component);
        AssertThrow (density > 0.0, ExcMessage ("Negative density encountered!"));
        const double pressure =
        EulerEquations<dim>::template compute_pressure<double>(solution_values[q]);
        AssertThrow (pressure > 0.0, ExcMessage ("Negative pressure encountered!"));

        const double sound_speed = std::sqrt(EulerEquations<dim>::gas_gamma * pressure/density);
        Tensor<1,dim> momentum;
        for (unsigned int i=EulerEquations<dim>::first_momentum_component;
             i < EulerEquations<dim>::first_momentum_component+dim; ++i)
          {
          momentum[i] = solution_values[q](i);
          }
        velocity = momentum.norm()/density;
        min_time_step = std::min(min_time_step,
                                 cell_size / (velocity+sound_speed) * parameters.CFL_number);
        }
      }
    parameters.time_step = min_time_step;
    }
  parameters.time_step *= parameters.time_step_factor;
  }

  // @sect4{ConservationLaw::assemble_system}
  //
  // This and the following two functions are the meat of this program: They
  // assemble the linear system that results from applying Newton's method to
  // the nonlinear system of conservation equations.
  //
  // This first function puts all of the assembly pieces together in a routine
  // that dispatches the correct piece for each cell/face.  The actual
  // implementation of the assembly on these objects is done in the following
  // functions.
  //
  // At the top of the function we do the usual housekeeping: allocate
  // FEValues, FEFaceValues, and FESubfaceValues objects necessary to do the
  // integrations on cells, faces, and subfaces (in case of adjoining cells on
  // different refinement levels). Note that we don't need all information
  // (like values, gradients, or real locations of quadrature points) for all
  // of these objects, so we only let the FEValues classes whatever is
  // actually necessary by specifying the minimal set of UpdateFlags. For
  // example, when using a FEFaceValues object for the neighboring cell we
  // only need the shape values: Given a specific face, the quadrature points
  // and <code>JxW</code> values are the same as for the current cells, and
  // the normal vectors are known to be the negative of the normal vectors of
  // the current cell.
  template <int dim>
  void ConservationLaw<dim>::assemble_system ()
  {
  const unsigned int dofs_per_cell = dof_handler.get_fe().dofs_per_cell;

  std::vector<types::global_dof_index> dof_indices (dofs_per_cell);
  std::vector<types::global_dof_index> dof_indices_neighbor (dofs_per_cell);

  const UpdateFlags update_flags               = update_values
  | update_gradients
  | update_q_points
  | update_JxW_values;
  const UpdateFlags face_update_flags          = update_values
  | update_q_points
  | update_JxW_values
  | update_normal_vectors;
  const UpdateFlags neighbor_face_update_flags = update_values;

  FEValues<dim>        fe_v                  (mapping, fe, quadrature,
                                              update_flags);
  FEFaceValues<dim>    fe_v_face             (mapping, fe, face_quadrature,
                                              face_update_flags);
  FESubfaceValues<dim> fe_v_subface          (mapping, fe, face_quadrature,
                                              face_update_flags);
  FEFaceValues<dim>    fe_v_face_neighbor    (mapping, fe, face_quadrature,
                                              neighbor_face_update_flags);
  FESubfaceValues<dim> fe_v_subface_neighbor (mapping, fe, face_quadrature,
                                              neighbor_face_update_flags);

  // Then loop over all cells, initialize the FEValues object for the
  // current cell and call the function that assembles the problem on this
  // cell.
  typename DoFHandler<dim>::active_cell_iterator
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();
  unsigned int cell_index(0);
  for (; cell!=endc; ++cell, ++cell_index)
    {
    fe_v.reinit (cell);
    cell->get_dof_indices (dof_indices);

    assemble_cell_term(fe_v, dof_indices, cell_index);

    // Then loop over all the faces of this cell.  If a face is part of
    // the external boundary, then assemble boundary conditions there (the
    // fifth argument to <code>assemble_face_terms</code> indicates
    // whether we are working on an external or internal face; if it is an
    // external face, the fourth argument denoting the degrees of freedom
    // indices of the neighbor is ignored, so we pass an empty vector):
    for (unsigned int face_no=0; face_no<GeometryInfo<dim>::faces_per_cell;
         ++face_no)
      if (cell->at_boundary(face_no))
        {
        fe_v_face.reinit (cell, face_no);
        assemble_face_term (face_no, fe_v_face,
                            fe_v_face,
                            dof_indices,
                            std::vector<types::global_dof_index>(),
                            true,
                            cell->face(face_no)->boundary_indicator(),
                            cell->face(face_no)->diameter());
        }

    // The alternative is that we are dealing with an internal face. There
    // are two cases that we need to distinguish: that this is a normal
    // face between two cells at the same refinement level, and that it is
    // a face between two cells of the different refinement levels.
    //
    // In the first case, there is nothing we need to do: we are using a
    // continuous finite element, and face terms do not appear in the
    // bilinear form in this case. The second case usually does not lead
    // to face terms either if we enforce hanging node constraints
    // strongly (as in all previous tutorial programs so far whenever we
    // used continuous finite elements -- this enforcement is done by the
    // ConstraintMatrix class together with
    // DoFTools::make_hanging_node_constraints). In the current program,
    // however, we opt to enforce continuity weakly at faces between cells
    // of different refinement level, for two reasons: (i) because we can,
    // and more importantly (ii) because we would have to thread the
    // automatic differentiation we use to compute the elements of the
    // Newton matrix from the residual through the operations of the
    // ConstraintMatrix class. This would be possible, but is not trivial,
    // and so we choose this alternative approach.
    //
    // What needs to be decided is which side of an interface between two
    // cells of different refinement level we are sitting on.
    //
    // Let's take the case where the neighbor is more refined first. We
    // then have to loop over the children of the face of the current cell
    // and integrate on each of them. We sprinkle a couple of assertions
    // into the code to ensure that our reasoning trying to figure out
    // which of the neighbor's children's faces coincides with a given
    // subface of the current cell's faces is correct -- a bit of
    // defensive programming never hurts.
    //
    // We then call the function that integrates over faces; since this is
    // an internal face, the fifth argument is false, and the sixth one is
    // ignored so we pass an invalid value again:
      else
        {
        if (cell->neighbor(face_no)->has_children())
          {
          const unsigned int neighbor2=
          cell->neighbor_of_neighbor(face_no);

          for (unsigned int subface_no=0;
               subface_no < cell->face(face_no)->n_children();
               ++subface_no)
            {
            const typename DoFHandler<dim>::active_cell_iterator
            neighbor_child
            = cell->neighbor_child_on_subface (face_no, subface_no);

            Assert (neighbor_child->face(neighbor2) ==
                    cell->face(face_no)->child(subface_no),
                    ExcInternalError());
            Assert (neighbor_child->has_children() == false,
                    ExcInternalError());

            fe_v_subface.reinit (cell, face_no, subface_no);
            fe_v_face_neighbor.reinit (neighbor_child, neighbor2);

            neighbor_child->get_dof_indices (dof_indices_neighbor);

            assemble_face_term (face_no, fe_v_subface,
                                fe_v_face_neighbor,
                                dof_indices,
                                dof_indices_neighbor,
                                false,
                                numbers::invalid_unsigned_int,
                                neighbor_child->face(neighbor2)->diameter());
            }
          }

        // The other possibility we have to care for is if the neighbor
        // is coarser than the current cell (in particular, because of
        // the usual restriction of only one hanging node per face, the
        // neighbor must be exactly one level coarser than the current
        // cell, something that we check with an assertion). Again, we
        // then integrate over this interface:
        else if (cell->neighbor(face_no)->level() != cell->level())
          {
          const typename DoFHandler<dim>::cell_iterator
          neighbor = cell->neighbor(face_no);
          Assert(neighbor->level() == cell->level()-1,
                 ExcInternalError());

          neighbor->get_dof_indices (dof_indices_neighbor);

          const std::pair<unsigned int, unsigned int>
          faceno_subfaceno = cell->neighbor_of_coarser_neighbor(face_no);
          const unsigned int neighbor_face_no    = faceno_subfaceno.first,
          neighbor_subface_no = faceno_subfaceno.second;

          Assert (neighbor->neighbor_child_on_subface (neighbor_face_no,
                                                       neighbor_subface_no)
                  == cell,
                  ExcInternalError());

          fe_v_face.reinit (cell, face_no);
          fe_v_subface_neighbor.reinit (neighbor,
                                        neighbor_face_no,
                                        neighbor_subface_no);

          assemble_face_term (face_no, fe_v_face,
                              fe_v_subface_neighbor,
                              dof_indices,
                              dof_indices_neighbor,
                              false,
                              numbers::invalid_unsigned_int,
                              cell->face(face_no)->diameter());
          }
        }
    }

  // After all this assembling, notify the Trilinos matrix object that the
  // matrix is done:
  system_matrix.compress(VectorOperation::add);
  }


  // @sect4{ConservationLaw::assemble_cell_term}
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
  ConservationLaw<dim>::
  assemble_cell_term (const FEValues<dim>             &fe_v,
                      const std::vector<types::global_dof_index> &dof_indices,
                      const unsigned int cell_index)
  {
  const unsigned int dofs_per_cell = fe_v.dofs_per_cell;
  const unsigned int n_q_points    = fe_v.n_quadrature_points;

  Table<2,Sacado::Fad::DFad<double> >
  W (n_q_points, EulerEquations<dim>::n_components);

  Table<2,double>
  W_old (n_q_points, EulerEquations<dim>::n_components);

  Table<3,Sacado::Fad::DFad<double> >
  grad_W (n_q_points, EulerEquations<dim>::n_components, dim);
  Table<3,double>
  grad_W_old(n_q_points, EulerEquations<dim>::n_components, dim);

  std::vector<double> residual_derivatives (dofs_per_cell);

  // Next, we have to define the independent variables that we will try to
  // determine by solving a Newton step. These independent variables are the
  // values of the local degrees of freedom which we extract here:
  std::vector<Sacado::Fad::DFad<double> > independent_local_dof_values(dofs_per_cell);
  for (unsigned int i=0; i<dofs_per_cell; ++i)
    {
    independent_local_dof_values[i] = current_solution(dof_indices[i]);
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
    for (unsigned int c=0; c<EulerEquations<dim>::n_components; ++c)
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
      const unsigned int c = fe_v.get_fe().system_to_component_index(i).first;

      W[q][c] += independent_local_dof_values[i] *
      fe_v.shape_value_component(i, q, c);
      W_old[q][c] += old_solution(dof_indices[i]) *
      fe_v.shape_value_component(i, q, c);

      for (unsigned int d = 0; d < dim; d++)
        {
        grad_W[q][c][d] += independent_local_dof_values[i] *
        fe_v.shape_grad_component(i, q, c)[d];
        grad_W_old[q][c][d] += old_solution(dof_indices[i]) *
        fe_v.shape_grad_component(i, q, c)[d];
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
  std::array <std::array <Sacado::Fad::DFad<double>, dim>, EulerEquations<dim>::n_components >
  > flux(n_q_points);

  std::vector <
  std::array <std::array <double, dim>, EulerEquations<dim>::n_components >
  > flux_old(n_q_points);

  std::vector < std::array< Sacado::Fad::DFad<double>, EulerEquations<dim>::n_components> > forcing(n_q_points);

  std::vector < std::array< double, EulerEquations<dim>::n_components> > forcing_old(n_q_points);

  for (unsigned int q=0; q<n_q_points; ++q)
    {
    EulerEquations<dim>::compute_flux_matrix (W_old[q], flux_old[q]);
    EulerEquations<dim>::compute_forcing_vector (W_old[q], forcing_old[q], parameters.gravity);
    EulerEquations<dim>::compute_flux_matrix (W[q], flux[q]);
    EulerEquations<dim>::compute_forcing_vector (W[q], forcing[q], parameters.gravity);
    }

  // evaluate viscosity coefficient
  //
  double viscos_coeff = 0.001;
  double rho_max(-1.0), D_h_max(-1.0), characteristic_speed_max(-1.0);

  for (unsigned int q=0; q<n_q_points; ++q)
    {
    // Here, we need to evaluate the derivatives of entropy flux respect to Euler equation independent variables $w$
    // rather than the unknown vector $W$. So we have to set up a new Sacado::Fad::DFad syetem.
    std::array<Sacado::Fad::DFad<double>, EulerEquations<dim>::n_components> w_for_entropy_flux;
    for (unsigned int c=0; c<EulerEquations<dim>::n_components; ++c)
      {
      w_for_entropy_flux[c] = W[q][c].val();
      w_for_entropy_flux[c].diff(c, EulerEquations<dim>::n_components);
      }

    const Sacado::Fad::DFad<double> entropy = EulerEquations<dim>::template compute_entropy<Sacado::Fad::DFad<double> >(w_for_entropy_flux);
    const double entroy_old = EulerEquations<dim>::template compute_entropy<double>(W_old[q]);

    double D_h1(0.0),D_h2(0.0);
    D_h1 = (entropy.val() - entroy_old)/parameters.time_step;
    D_h2 = (W[q][EulerEquations<dim>::density_component].val() - W_old[q][EulerEquations<dim>::density_component])/
    parameters.time_step;

    //sum up divergence
    for (unsigned int d=0; d<dim; d++)
      {
      const Sacado::Fad::DFad<double> entropy_flux = entropy *
      w_for_entropy_flux[EulerEquations<dim>::first_momentum_component + d]/
      w_for_entropy_flux[EulerEquations<dim>::density_component];
      for (unsigned int c=0; c<EulerEquations<dim>::n_components; ++c)
        {
        D_h1 += entropy_flux.fastAccessDx(c) * grad_W[q][c][d].val();
        }
      D_h2 += grad_W[q][EulerEquations<dim>::first_momentum_component + d][d].val();
      }
    D_h2 *= entropy.val()/W[q][EulerEquations<dim>::density_component].val();
    D_h_max = std::max(D_h_max, std::abs(D_h1));
    D_h_max = std::max(D_h_max, std::abs(D_h2));

    rho_max = std::max(rho_max, W[q][EulerEquations<dim>::density_component].val());

    // Calculate local sound speed.
    const double density = W[q][EulerEquations<dim>::density_component].val();
    AssertThrow (density > 0.0, ExcMessage ("Negative density encountered!"));
    const double pressure =
    EulerEquations<dim>::template compute_pressure<Sacado::Fad::DFad<double> >(W[q]).val();
    AssertThrow (pressure > 0.0, ExcMessage ("Negative pressure encountered!"));

    const double sound_speed = std::sqrt(EulerEquations<dim>::gas_gamma * pressure/density);

    // Calculate local velosity magnitude.
    Tensor<1,dim> momentum;
    for (unsigned int i=EulerEquations<dim>::first_momentum_component;
         i < EulerEquations<dim>::first_momentum_component+dim; ++i)
      {
      momentum[i] = W[q][i].val();
      }
    const double velocity = momentum.norm()/density;

    characteristic_speed_max = std::max(characteristic_speed_max, velocity + sound_speed);
    }
  const double cE = 1.0;
  const double entropy_visc = cE * rho_max * std::pow(fe_v.get_cell()->diameter(), 1.5) * D_h_max;
  const double cMax = 0.25;
  const double miu_max = cMax * fe_v.get_cell()->diameter() * rho_max * characteristic_speed_max;
  //  const double miu_max = 1.0*std::pow(fe_v.get_cell()->diameter(), parameters.diffusion_power);

  entropy_viscosity[cell_index] = std::min(miu_max, entropy_visc);
  //  entropy_viscosity[cell_index] = std::max(0.002, entropy_viscosity[cell_index]);
  //


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
  // t},(\mathbf{z}_i)_{\text{component\_i}}\right)_K \\
  // &-& \sum_{d=1}^{\text{dim}} \left(  \theta \mathbf{F}
  // ({\mathbf{w^k_{n+1}}})_{\text{component\_i},d} + (1-\theta)
  // \mathbf{F} ({\mathbf{w_{n}}})_{\text{component\_i},d}  ,
  // \frac{\partial(\mathbf{z}_i)_{\text{component\_i}}} {\partial
  // x_d}\right)_K \\
  // &+& \sum_{d=1}^{\text{dim}} h^{\eta} \left( \theta \frac{\partial
  // \mathbf{w^k_{n+1}}_{\text{component\_i}}}{\partial x_d} + (1-\theta)
  // \frac{\partial \mathbf{w_n}_{\text{component\_i}}}{\partial x_d} ,
  // \frac{\partial (\mathbf{z}_i)_{\text{component\_i}}}{\partial x_d} \right)_K\\
  // &-& \left( \theta\mathbf{G}({\mathbf{w}^k_n+1} )_{\text{component\_i}} +
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
  for (unsigned int i=0; i<fe_v.dofs_per_cell; ++i)
    {
    Sacado::Fad::DFad<double> R_i = 0;

    const unsigned int
    component_i = fe_v.get_fe().system_to_component_index(i).first;

    // The residual for each row (i) will be accumulating into this fad
    // variable.  At the end of the assembly for this row, we will query
    // for the sensitivities to this variable and add them into the
    // Jacobian.

    for (unsigned int point=0; point<fe_v.n_quadrature_points; ++point)
      {
      if (parameters.is_stationary == false)
        R_i += 1.0 / parameters.time_step *
        (W[point][component_i] - W_old[point][component_i]) *
        fe_v.shape_value_component(i, point, component_i) *
        fe_v.JxW(point);

      for (unsigned int d=0; d<dim; d++)
        R_i -= (parameters.theta * flux[point][component_i][d] +
                (1.0-parameters.theta) * flux_old[point][component_i][d]) *
        fe_v.shape_grad_component(i, point, component_i)[d] *
        fe_v.JxW(point);

      cellSize_viscosity[cell_index] = 1.0*std::pow(fe_v.get_cell()->diameter(), parameters.diffusion_power);

      viscos_coeff = 0.000;
      if(parameters.diffusion_type == Parameters::AllParameters<dim>::diffu_entropy)
        {
        viscos_coeff = entropy_viscosity[cell_index];
        }
      if(parameters.diffusion_type == Parameters::AllParameters<dim>::diffu_cell_size)
        {
        viscos_coeff = cellSize_viscosity[cell_index];
        }
      if(parameters.diffusion_type == Parameters::AllParameters<dim>::diffu_const)
        {
        viscos_coeff = parameters.diffusion_coefficoent;
        }
      for (unsigned int d=0; d<dim; d++)
        {
        R_i += viscos_coeff *
        (parameters.theta * grad_W[point][component_i][d] +
         (1.0-parameters.theta) * grad_W_old[point][component_i][d]) *
        fe_v.shape_grad_component(i, point, component_i)[d] *
        fe_v.JxW(point);
        }

      R_i -= (parameters.theta  * forcing[point][component_i] +
              (1.0 - parameters.theta) * forcing_old[point][component_i]) *
      fe_v.shape_value_component(i, point, component_i) *
      fe_v.JxW(point);
      }

    // At the end of the loop, we have to add the sensitivities to the
    // matrix and subtract the residual from the right hand side. Trilinos
    // FAD data type gives us access to the derivatives using
    // <code>R_i.fastAccessDx(k)</code>, so we store the data in a
    // temporary array. This information about the whole row of local dofs
    // is then added to the Trilinos matrix at once (which supports the
    // data types we have chosen).
    for (unsigned int k=0; k<dofs_per_cell; ++k)
      {
      residual_derivatives[k] = R_i.fastAccessDx(k);
      }
    system_matrix.add(dof_indices[i], dof_indices, residual_derivatives);
    right_hand_side(dof_indices[i]) -= R_i.val();
    }
  }


  // @sect4{ConservationLaw::assemble_face_term}
  //
  // Here, we do essentially the same as in the previous function. At the top,
  // we introduce the independent variables. Because the current function is
  // also used if we are working on an internal face between two cells, the
  // independent variables are not only the degrees of freedom on the current
  // cell but in the case of an interior face also the ones on the neighbor.
  template <int dim>
  void
  ConservationLaw<dim>::assemble_face_term(const unsigned int           face_no,
                                           const FEFaceValuesBase<dim> &fe_v,
                                           const FEFaceValuesBase<dim> &fe_v_neighbor,
                                           const std::vector<types::global_dof_index>   &dof_indices,
                                           const std::vector<types::global_dof_index>   &dof_indices_neighbor,
                                           const bool                   external_face,
                                           const unsigned int           boundary_id,
                                           const double                 face_diameter)
  {
  const unsigned int n_q_points = fe_v.n_quadrature_points;
  const unsigned int dofs_per_cell = fe_v.dofs_per_cell;

  std::vector<Sacado::Fad::DFad<double> >
  independent_local_dof_values (dofs_per_cell),
  independent_neighbor_dof_values (external_face == false ?
                                   dofs_per_cell :
                                   0);

  const unsigned int n_independent_variables = (external_face == false ?
                                                2 * dofs_per_cell :
                                                dofs_per_cell);

  for (unsigned int i = 0; i < dofs_per_cell; i++)
    {
    independent_local_dof_values[i] = current_solution(dof_indices[i]);
    independent_local_dof_values[i].diff(i, n_independent_variables);
    }

  if (external_face == false)
    for (unsigned int i = 0; i < dofs_per_cell; i++)
      {
      independent_neighbor_dof_values[i]
      = current_solution(dof_indices_neighbor[i]);
      independent_neighbor_dof_values[i]
      .diff(i+dofs_per_cell, n_independent_variables);
      }


  // Next, we need to define the values of the conservative variables
  // ${\mathbf W}$ on this side of the face ($ {\mathbf W}^+$)
  // and on the opposite side (${\mathbf W}^-$), for both ${\mathbf W} =
  // {\mathbf W}^k_{n+1}$ and  ${\mathbf W} = {\mathbf W}_n$.
  // The "this side" values can be
  // computed in exactly the same way as in the previous function, but note
  // that the <code>fe_v</code> variable now is of type FEFaceValues or
  // FESubfaceValues:
  Table<2,Sacado::Fad::DFad<double> >
  Wplus (n_q_points, EulerEquations<dim>::n_components),
  Wminus (n_q_points, EulerEquations<dim>::n_components);
  Table<2,double>
  Wplus_old(n_q_points, EulerEquations<dim>::n_components),
  Wminus_old(n_q_points, EulerEquations<dim>::n_components);

  for (unsigned int q=0; q<n_q_points; ++q)
    for (unsigned int i=0; i<dofs_per_cell; ++i)
      {
      const unsigned int component_i = fe_v.get_fe().system_to_component_index(i).first;
      Wplus[q][component_i] +=  independent_local_dof_values[i] *
      fe_v.shape_value_component(i, q, component_i);
      Wplus_old[q][component_i] +=  old_solution(dof_indices[i]) *
      fe_v.shape_value_component(i, q, component_i);
      }

  // Computing "opposite side" is a bit more complicated. If this is
  // an internal face, we can compute it as above by simply using the
  // independent variables from the neighbor:
  if (external_face == false)
    {
    for (unsigned int q=0; q<n_q_points; ++q)
      for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
        const unsigned int component_i = fe_v_neighbor.get_fe().
        system_to_component_index(i).first;
        Wminus[q][component_i] += independent_neighbor_dof_values[i] *
        fe_v_neighbor.shape_value_component(i, q, component_i);
        Wminus_old[q][component_i] += old_solution(dof_indices_neighbor[i])*
        fe_v_neighbor.shape_value_component(i, q, component_i);
        }
    }
  // On the other hand, if this is an external boundary face, then the
  // values of $W^-$ will be either functions of $W^+$, or they will be
  // prescribed, depending on the kind of boundary condition imposed here.
  //
  // To start the evaluation, let us ensure that the boundary id specified
  // for this boundary is one for which we actually have data in the
  // parameters object. Next, we evaluate the function object for the
  // inhomogeneity.  This is a bit tricky: a given boundary might have both
  // prescribed and implicit values.  If a particular component is not
  // prescribed, the values evaluate to zero and are ignored below.
  //
  // The rest is done by a function that actually knows the specifics of
  // Euler equation boundary conditions. Note that since we are using fad
  // variables here, sensitivities will be updated appropriately, a process
  // that would otherwise be tremendously complicated.
  else
    {
    Assert (boundary_id < Parameters::AllParameters<dim>::max_n_boundaries,
            ExcIndexRange (boundary_id, 0,
                           Parameters::AllParameters<dim>::max_n_boundaries));

    std::vector<Vector<double> >
    boundary_values(n_q_points, Vector<double>(EulerEquations<dim>::n_components));
    parameters.boundary_conditions[boundary_id]
    .values.vector_value_list(fe_v.get_quadrature_points(),
                              boundary_values);

    for (unsigned int q = 0; q < n_q_points; q++)
      {
      EulerEquations<dim>::compute_Wminus (parameters.boundary_conditions[boundary_id].kind,
                                           fe_v.normal_vector(q),
                                           Wplus[q],
                                           boundary_values[q],
                                           Wminus[q]);
      // Here we assume that boundary type, boundary normal vector and boundary data values
      // maintain the same during time advancing.
      EulerEquations<dim>::compute_Wminus (parameters.boundary_conditions[boundary_id].kind,
                                           fe_v.normal_vector(q),
                                           Wplus_old[q],
                                           boundary_values[q],
                                           Wminus_old[q]);
      }
    }


  // Now that we have $\mathbf w^+$ and $\mathbf w^-$, we can go about
  // computing the numerical flux function $\mathbf H(\mathbf w^+,\mathbf
  // w^-, \mathbf n)$ for each quadrature point. Before calling the function
  // that does so, we also need to determine the Lax-Friedrich's stability
  // parameter:
  std::vector< std::array < Sacado::Fad::DFad<double>, EulerEquations<dim>::n_components> >  normal_fluxes(n_q_points);
  std::vector< std::array < double, EulerEquations<dim>::n_components> >  normal_fluxes_old(n_q_points);


  double alpha;

  switch (parameters.stabilization_kind)
    {
      case Parameters::Flux::constant:
      alpha = parameters.stabilization_value;
      break;
      case Parameters::Flux::mesh_dependent:
      alpha = face_diameter/(2.0*parameters.time_step);
      break;
      default:
      Assert (false, ExcNotImplemented());
      alpha = 1;
    }

  for (unsigned int q=0; q<n_q_points; ++q)
    {
    EulerEquations<dim>::numerical_normal_flux(fe_v.normal_vector(q),
                                               Wplus[q], Wminus[q], alpha,
                                               normal_fluxes[q]);
    EulerEquations<dim>::numerical_normal_flux(fe_v.normal_vector(q),
                                               Wplus_old[q], Wminus_old[q], alpha,
                                               normal_fluxes_old[q]);
    }

  // Now assemble the face term in exactly the same way as for the cell
  // contributions in the previous function. The only difference is that if
  // this is an internal face, we also have to take into account the
  // sensitivities of the residual contributions to the degrees of freedom on
  // the neighboring cell:
  std::vector<double> residual_derivatives (dofs_per_cell);
  for (unsigned int i=0; i<fe_v.dofs_per_cell; ++i)
    if (fe_v.get_fe().has_support_on_face(i, face_no) == true)
      {
      Sacado::Fad::DFad<double> R_i = 0;

      for (unsigned int point=0; point<n_q_points; ++point)
        {
        const unsigned int
        component_i = fe_v.get_fe().system_to_component_index(i).first;

        R_i += (parameters.theta * normal_fluxes[point][component_i] +
                (1.0 - parameters.theta) * normal_fluxes_old[point][component_i]) *
        fe_v.shape_value_component(i, point, component_i) *
        fe_v.JxW(point);
        }

      for (unsigned int k=0; k<dofs_per_cell; ++k)
        {
        residual_derivatives[k] = R_i.fastAccessDx(k);
        }
      system_matrix.add(dof_indices[i], dof_indices, residual_derivatives);

      if (external_face == false)
        {
        for (unsigned int k=0; k<dofs_per_cell; ++k)
          {
          residual_derivatives[k] = R_i.fastAccessDx(dofs_per_cell+k);
          }
        system_matrix.add (dof_indices[i], dof_indices_neighbor,
                           residual_derivatives);
        }

      right_hand_side(dof_indices[i]) -= R_i.val();
      }
  }


  // @sect4{ConservationLaw::solve}
  //
  // Here, we actually solve the linear system, using either of Trilinos'
  // Aztec or Amesos linear solvers. The result of the computation will be
  // written into the argument vector passed to this function. The result is a
  // pair of number of iterations and the final linear residual.

  template <int dim>
  std::pair<unsigned int, double>
  ConservationLaw<dim>::solve (Vector<double> &newton_update)
  {
  switch (parameters.solver)
    {
      // If the parameter file specified that a direct solver shall be used,
      // then we'll get here. The process is straightforward, since deal.II
      // provides a wrapper class to the Amesos direct solver within
      // Trilinos. All we have to do is to create a solver control object
      // (which is just a dummy object here, since we won't perform any
      // iterations), and then create the direct solver object. When
      // actually doing the solve, note that we don't pass a
      // preconditioner. That wouldn't make much sense for a direct solver
      // anyway.  At the end we return the solver control statistics &mdash;
      // which will tell that no iterations have been performed and that the
      // final linear residual is zero, absent any better information that
      // may be provided here:
      case Parameters::Solver::direct:
      {
      SolverControl solver_control (1,0);
      TrilinosWrappers::SolverDirect direct (solver_control,
                                             parameters.output ==
                                             Parameters::Solver::verbose);

      direct.solve (system_matrix, newton_update, right_hand_side);

      return std::pair<unsigned int, double> (solver_control.last_step(),
                                              solver_control.last_value());
      }

      // Likewise, if we are to use an iterative solver, we use Aztec's GMRES
      // solver. We could use the Trilinos wrapper classes for iterative
      // solvers and preconditioners here as well, but we choose to use an
      // Aztec solver directly. For the given problem, Aztec's internal
      // preconditioner implementations are superior over the ones deal.II has
      // wrapper classes to, so we use ILU-T preconditioning within the
      // AztecOO solver and set a bunch of options that can be changed from
      // the parameter file.
      //
      // There are two more practicalities: Since we have built our right hand
      // side and solution vector as deal.II Vector objects (as opposed to the
      // matrix, which is a Trilinos object), we must hand the solvers
      // Trilinos Epetra vectors.  Luckily, they support the concept of a
      // 'view', so we just send in a pointer to our deal.II vectors. We have
      // to provide an Epetra_Map for the vector that sets the parallel
      // distribution, which is just a dummy object in serial. The easiest way
      // is to ask the matrix for its map, and we're going to be ready for
      // matrix-vector products with it.
      //
      // Secondly, the Aztec solver wants us to pass a Trilinos
      // Epetra_CrsMatrix in, not the deal.II wrapper class itself. So we
      // access to the actual Trilinos matrix in the Trilinos wrapper class by
      // the command trilinos_matrix(). Trilinos wants the matrix to be
      // non-constant, so we have to manually remove the constantness using a
      // const_cast.
      case Parameters::Solver::gmres:
      {
      Epetra_Vector x(View, system_matrix.domain_partitioner(),
                      newton_update.begin());
      Epetra_Vector b(View, system_matrix.range_partitioner(),
                      right_hand_side.begin());

      AztecOO solver;
      solver.SetAztecOption(AZ_output,
                            (parameters.output ==
                             Parameters::Solver::quiet
                             ?
                             AZ_none
                             :
                             AZ_all));
      solver.SetAztecOption(AZ_solver, AZ_gmres);
      solver.SetRHS(&b);
      solver.SetLHS(&x);

      solver.SetAztecOption(AZ_precond,         AZ_dom_decomp);
      solver.SetAztecOption(AZ_subdomain_solve, AZ_ilut);
      solver.SetAztecOption(AZ_overlap,         0);
      solver.SetAztecOption(AZ_reorder,         0);

      solver.SetAztecParam(AZ_drop,      parameters.ilut_drop);
      solver.SetAztecParam(AZ_ilut_fill, parameters.ilut_fill);
      solver.SetAztecParam(AZ_athresh,   parameters.ilut_atol);
      solver.SetAztecParam(AZ_rthresh,   parameters.ilut_rtol);

      solver.SetUserMatrix(const_cast<Epetra_CrsMatrix *>
                           (&system_matrix.trilinos_matrix()));

      solver.Iterate(parameters.max_iterations, parameters.linear_residual);

      return std::pair<unsigned int, double> (solver.NumIters(),
                                              solver.TrueResidual());
      }
    }

  Assert (false, ExcNotImplemented());
  return std::pair<unsigned int, double> (0,0);
  }


  // @sect4{ConservationLaw::compute_refinement_indicators}

  // This function is real simple: We don't pretend that we know here what a
  // good refinement indicator would be. Rather, we assume that the
  // <code>EulerEquation</code> class would know about this, and so we simply
  // defer to the respective function we've implemented there:
  template <int dim>
  void
  ConservationLaw<dim>::
  compute_refinement_indicators (Vector<double> &refinement_indicators) const
  {
  EulerEquations<dim>::compute_refinement_indicators (dof_handler,
                                                      mapping,
                                                      predictor,
                                                      refinement_indicators);
  }



  // @sect4{ConservationLaw::refine_grid}

  // Here, we use the refinement indicators computed before and refine the
  // mesh. At the beginning, we loop over all cells and mark those that we
  // think should be refined:
  template <int dim>
  void
  ConservationLaw<dim>::refine_grid (const Vector<double> &refinement_indicators)
  {
  typename DoFHandler<dim>::active_cell_iterator
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();

  for (unsigned int cell_no=0; cell!=endc; ++cell, ++cell_no)
    {
    cell->clear_coarsen_flag();
    cell->clear_refine_flag();

    if ((cell->level() < parameters.shock_levels) &&
        (std::fabs(refinement_indicators(cell_no)) > parameters.shock_val))
      {
      cell->set_refine_flag();
      }
    else if ((cell->level() > 0) &&
             (std::fabs(refinement_indicators(cell_no)) < 0.75*parameters.shock_val))
      {
      cell->set_coarsen_flag();
      }
    }

  // Then we need to transfer the various solution vectors from the old to
  // the new grid while we do the refinement. The SolutionTransfer class is
  // our friend here; it has a fairly extensive documentation, including
  // examples, so we won't comment much on the following code. The last
  // three lines simply re-set the sizes of some other vectors to the now
  // correct size:
  std::vector<Vector<double> > transfer_in;
  std::vector<Vector<double> > transfer_out;

  transfer_in.push_back(old_solution);
  transfer_in.push_back(predictor);

  triangulation.prepare_coarsening_and_refinement();

  SolutionTransfer<dim> soltrans(dof_handler);
  soltrans.prepare_for_coarsening_and_refinement(transfer_in);

  triangulation.execute_coarsening_and_refinement ();

  dof_handler.clear();
  dof_handler.distribute_dofs (fe);

    {
    Vector<double> new_old_solution(1);
    Vector<double> new_predictor(1);

    transfer_out.push_back(new_old_solution);
    transfer_out.push_back(new_predictor);
    transfer_out[0].reinit(dof_handler.n_dofs());
    transfer_out[1].reinit(dof_handler.n_dofs());
    }

  soltrans.interpolate(transfer_in, transfer_out);

  old_solution.reinit (transfer_out[0].size());
  old_solution = transfer_out[0];

  predictor.reinit (transfer_out[1].size());
  predictor = transfer_out[1];

  current_solution.reinit(dof_handler.n_dofs());
  current_solution = old_solution;
  right_hand_side.reinit (dof_handler.n_dofs());
  current_solution_backup.reinit (dof_handler.n_dofs());

  entropy_viscosity.reinit(triangulation.n_active_cells());
  cellSize_viscosity.reinit(triangulation.n_active_cells());
  }


  // @sect4{ConservationLaw::output_results}

  // This function now is rather straightforward. All the magic, including
  // transforming data from conservative variables to physical ones has been
  // abstracted and moved into the EulerEquations class so that it can be
  // replaced in case we want to solve some other hyperbolic conservation law.
  //
  // Note that the number of the output file is determined by keeping a
  // counter in the form of a static variable that is set to zero the first
  // time we come to this function and is incremented by one at the end of
  // each invocation.
  template <int dim>
  void ConservationLaw<dim>::output_results () const
  {
  typename EulerEquations<dim>::Postprocessor
  postprocessor (parameters.schlieren_plot);

  DataOut<dim> data_out;
  data_out.attach_dof_handler (dof_handler);

  data_out.add_data_vector (current_solution,
                            EulerEquations<dim>::component_names (),
                            DataOut<dim>::type_dof_data,
                            EulerEquations<dim>::component_interpretation ());

  data_out.add_data_vector (current_solution, postprocessor);

    {
    const std::string data_name("entropy_viscosity");
    data_out.add_data_vector (entropy_viscosity,
                              data_name,
                              DataOut<dim>::type_cell_data);
    }
    {
    const std::string data_name("cellSize_viscosity");
    data_out.add_data_vector (cellSize_viscosity,
                              data_name,
                              DataOut<dim>::type_cell_data);
    }

  data_out.build_patches ();

  static unsigned int output_file_number = 0;
  std::string filename = "solution-" +
  Utilities::int_to_string (output_file_number, 3) +
  ".vtk";
  std::ofstream output (filename.c_str());
  data_out.write_vtk (output);

  ++output_file_number;
  }




  // @sect4{ConservationLaw::run}

  // This function contains the top-level logic of this program:
  // initialization, the time loop, and the inner Newton iteration.
  //
  // At the beginning, we read the mesh file specified by the parameter file,
  // setup the DoFHandler and various vectors, and then interpolate the given
  // initial conditions on this mesh. We then perform a number of mesh
  // refinements, based on the initial conditions, to obtain a mesh that is
  // already well adapted to the starting solution. At the end of this
  // process, we output the initial solution.
  template <int dim>
  void ConservationLaw<dim>::run ()
  {
    {
    GridIn<dim> grid_in;
    grid_in.attach_triangulation(triangulation);

    std::ifstream input_file(parameters.mesh_filename.c_str());
    Assert (input_file, ExcFileNotOpen(parameters.mesh_filename.c_str()));

    if (parameters.mesh_format == Parameters::AllParameters<dim>::format_ucd)
      {
      grid_in.read_ucd(input_file);
      }
    if (parameters.mesh_format == Parameters::AllParameters<dim>::format_gmsh)
      {
      grid_in.read_msh(input_file); //QiaoL
      GridTools::scale(0.001,triangulation);
      }
    }

  dof_handler.clear();
  dof_handler.distribute_dofs (fe);

  // Size all of the fields.
  old_solution.reinit (dof_handler.n_dofs());
  current_solution.reinit (dof_handler.n_dofs());
  current_solution_backup.reinit(dof_handler.n_dofs());
  predictor.reinit (dof_handler.n_dofs());
  right_hand_side.reinit (dof_handler.n_dofs());

  entropy_viscosity.reinit(triangulation.n_active_cells());
  cellSize_viscosity.reinit(triangulation.n_active_cells());

  setup_system();

  VectorTools::interpolate(dof_handler,
                           parameters.initial_conditions, old_solution);
  current_solution = old_solution;
  predictor = old_solution;

  calc_time_step();

  if (parameters.do_refine == true)
    for (unsigned int i=0; i<parameters.shock_levels; ++i)
      {
      Vector<double> refinement_indicators (triangulation.n_active_cells());

      compute_refinement_indicators(refinement_indicators);
      refine_grid(refinement_indicators);

      setup_system();

      VectorTools::interpolate(dof_handler,
                               parameters.initial_conditions, old_solution);
      current_solution = old_solution;
      predictor = old_solution;
      }

  output_results ();

  // We then enter into the main time stepping loop. At the top we simply
  // output some status information so one can keep track of where a
  // computation is, as well as the header for a table that indicates
  // progress of the nonlinear inner iteration:
  Vector<double> newton_update (dof_handler.n_dofs());

  double time = 0;
  double next_output = time + parameters.output_step;

  predictor = old_solution;

  bool newton_iter_converged(false);
  bool time_step_doubled(false);
  int index_linear_search_length(0);
  const double linear_search_length[12]= {1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.025, 0.0125, 1.2, 1.5, 2.0};
  unsigned int converged_newton_iters(0);

  while (time < parameters.final_time)
    {
    std::cout << "T=" << time << std::endl
    << "   Number of active cells:       "
    << triangulation.n_active_cells()
    << std::endl
    << "   Number of degrees of freedom: "
    << dof_handler.n_dofs()
    << std::endl
    << std::endl;

    std::cout << "   NonLin Res     Lin Iter       Lin Res     "
    << "Linear Search Len      Time Step Size      Time Step Factor" << std::endl
    << "   __________________________________________"
    << "_____________________________________________" << std::endl;

    // Then comes the inner Newton iteration to solve the nonlinear
    // problem in each time step. The way it works is to reset matrix and
    // right hand side to zero, then assemble the linear system. If the
    // norm of the right hand side is small enough, then we declare that
    // the Newton iteration has converged. Otherwise, we solve the linear
    // system, update the current solution with the Newton increment, and
    // output convergence information. At the end, we check that the
    // number of Newton iterations is not beyond a limit of 10 -- if it
    // is, it appears likely that iterations are diverging and further
    // iterations would do no good. If that happens, we throw an exception
    // that will be caught in <code>main()</code> with status information
    // being displayed before the program aborts.
    //
    // Note that the way we write the AssertThrow macro below is by and
    // large equivalent to writing something like <code>if (!(nonlin_iter
    // @<= 10)) throw ExcMessage ("No convergence in nonlinear
    // solver");</code>. The only significant difference is that
    // AssertThrow also makes sure that the exception being thrown carries
    // with it information about the location (file name and line number)
    // where it was generated. This is not overly critical here, because
    // there is only a single place where this sort of exception can
    // happen; however, it is generally a very useful tool when one wants
    // to find out where an error occurred.

    double predictor_leap_ratio = 1.0; // Instead of always extrapolating predictor to the next
                                       // time step, we can make the forward extrapolation at
                                       // an adjustable ratio.
    newton_iter_converged = false;
    current_solution_backup = current_solution;

    unsigned int nonlin_iter = 0;
    current_solution = predictor;
    bool linear_solver_diverged(true);

    while (true)  //Begine Newton iteration
      {
      system_matrix = 0;

      right_hand_side = 0;
      assemble_system ();

      const double res_norm = right_hand_side.l2_norm();
      if (std::fabs(res_norm) < 1e-10)
        {
        std::printf("   %-16.3e (converged)\n\n", res_norm);
        newton_iter_converged = true;
        break;
        }
      else
        {
        newton_update = 0;

        std::pair<unsigned int, double> convergence
        = solve (newton_update);
        Assert(index_linear_search_length < 9, ExcIndexRange(index_linear_search_length,0,9));
        newton_update *= linear_search_length[index_linear_search_length];
        current_solution += newton_update;

        std::printf("   %-16.3e %04d        %-5.2e            %7.4g          %7.4g          %7.4g\n",
                    res_norm, convergence.first, convergence.second,
                    linear_search_length[index_linear_search_length],
                    parameters.time_step, parameters.time_step_factor);
        linear_solver_diverged = std::isnan(convergence.second);
        }
      ++nonlin_iter;
      const unsigned int nonlin_iter_threshold(10);
      if (linear_solver_diverged)
        {
        nonlin_iter = nonlin_iter_threshold + 1;
        std::cout << "  Linear solver diverged..\n";
        }

      if (nonlin_iter > nonlin_iter_threshold)
        {
        std::cout << "  Newton iteration not converge in " << nonlin_iter_threshold << " steps.\n"
        << "  Recompute with different linear search length or time step...\n\n";
        newton_iter_converged = false;
        break;
        }
      //            AssertThrow (nonlin_iter <= 10,
      //                         ExcMessage ("No convergence in nonlinear solver"));
      }

    if (newton_iter_converged)
      {
      // We only get to this point if the Newton iteration has converged, so
      // do various post convergence tasks here:
      //
      // First, we update the time and produce graphical output if so
      // desired. Then we update a predictor for the solution at the next
      // time step by approximating $\mathbf w^{n+1}\approx \mathbf w^n +
      // \delta t \frac{\partial \mathbf w}{\partial t} \approx \mathbf w^n
      // + \delta t \; \frac{\mathbf w^n-\mathbf w^{n-1}}{\delta t} = 2
      // \mathbf w^n - \mathbf w^{n-1}$ to try and make adaptivity work
      // better.  The idea is to try and refine ahead of a front, rather
      // than stepping into a coarse set of elements and smearing the
      // old_solution.  This simple time extrapolator does the job. With
      // this, we then refine the mesh if so desired by the user, and
      // finally continue on with the next time step:
      ++converged_newton_iters;
      time += parameters.time_step;

      if (parameters.output_step < 0)
        {
        output_results ();
        }
      else if (time >= next_output)
        {
        output_results ();
        next_output += parameters.output_step;
        }

      predictor = current_solution;

      if ( parameters.allow_double_time_step && converged_newton_iters%10 == 0 )
        {

        //Since every thing goes so well, let's try a larger time step next.
        parameters.time_step_factor *= 2.0;
        time_step_doubled = true;
        index_linear_search_length = 0;
        std::cout << "  We got ten successive converged time steps.\n"
        << "  Time step size increased to " << parameters.time_step << "\n\n";
        predictor.sadd (1.0+predictor_leap_ratio*2.0, 0.0-predictor_leap_ratio*2.0, old_solution);
        }
      else
        {
        predictor.sadd (1.0+predictor_leap_ratio,     0.0-predictor_leap_ratio,     old_solution);
        }

      // old_solution is going to be overwritten immediately.
      // Just use it to calculate the time advancing norms.
      old_solution.sadd (-1.0, current_solution);
      std::cout << "  Order of time advancing L_infty norm = "
      << std::log(old_solution.linfty_norm())/std::log(10.0) << std::endl;
      std::cout << "  Order of time advancing L_2     norm = "
      << std::log(old_solution.l2_norm())/std::log(10.0) << std::endl;

      old_solution = current_solution;

      if (parameters.do_refine == true)
        {
        Vector<double> refinement_indicators (triangulation.n_active_cells());
        compute_refinement_indicators(refinement_indicators);

        refine_grid(refinement_indicators);
        setup_system();

        newton_update.reinit (dof_handler.n_dofs());
        }
      current_solution_backup = current_solution;
      calc_time_step();
      // Uncomment the following line if you want reset the linear_search_length immediatly after a converged Newton iter.
      //index_linear_search_length = 0;
      }
    else
      {
      // Newton iteration not converge in reasonable steps
      if (index_linear_search_length < 8 && (!time_step_doubled))
        {
        // Try to adjust linear_search_length first
        ++index_linear_search_length;
        }
      else
        {
        // Reduce time step when linear_search_length has tried out.
        parameters.time_step *= 0.5;
        parameters.time_step_factor *= 0.5;
        time_step_doubled = false;
        index_linear_search_length = 0;
        std::cout << "  Time step size reduced to " << parameters.time_step << "\n\n";
        }

      current_solution = current_solution_backup;
      predictor = current_solution;
      if (converged_newton_iters > 0)
        {
        // The last good "current_solution" is calculated with a "large" time step,
        // so we need to make a near extrapolation.
        predictor.sadd (1.0+predictor_leap_ratio*0.5, 0.0-predictor_leap_ratio*0.5, old_solution);
        }
      else
        {
        predictor.sadd (1.0+predictor_leap_ratio, 0.0-predictor_leap_ratio, old_solution);
        }
      converged_newton_iters = 0;
      }
    }
  }

  template class ConservationLaw<2>;
  template class ConservationLaw<3>;
}

