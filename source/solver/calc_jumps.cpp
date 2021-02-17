//
//  NSolver::calc_jumps.cpp
//
//  Created by Lei Qiao on 15/11/6.
//  A work based on deal.II tutorial step-33.
//

#include <NSolver/solver/NSolver.h>

namespace NSFEMSolver
{
  using namespace dealii;
  template <int dim>
  double
  NSolver<dim>::calc_jumps(const FEFaceValuesBase<dim> &fe_v_face_this,
                           const FEFaceValuesBase<dim> &fe_v_face_neighbor,
                           const unsigned int           boundary_id,
                           const bool                   accumulate_grad_jump,
                           const bool                   accumulate_value_jump)
  {
    const unsigned int n_q_points = fe_v_face_this.n_quadrature_points;

    std::array<double, EquationComponents<dim>::n_components> max_jump;
    max_jump.fill(0.0);

    if (accumulate_grad_jump)
      {
        std::vector<std::vector<Tensor<1, dim>>> grad_W(
          n_q_points,
          std::vector<Tensor<1, dim>>(EquationComponents<dim>::n_components));

        std::vector<std::vector<Tensor<1, dim>>> grad_W_neighbor(
          n_q_points,
          std::vector<Tensor<1, dim>>(EquationComponents<dim>::n_components));

        std::array<double, EquationComponents<dim>::n_components> max_grad_jump;

        fe_v_face_this.get_function_gradients(current_solution, grad_W);
        fe_v_face_neighbor.get_function_gradients(current_solution,
                                                  grad_W_neighbor);
        max_grad_jump.fill(0.0);
        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            const Tensor<1, dim> normal_vector =
              fe_v_face_this.normal_vector(q);
            for (unsigned int c = 0; c < EquationComponents<dim>::n_components;
                 ++c)
              {
                max_grad_jump[c] =
                  std::max(max_grad_jump[c],
                           std::abs(normal_vector *
                                    (grad_W[q][c] - grad_W_neighbor[q][c])));
              }
          }
        for (unsigned int c = 0; c < EquationComponents<dim>::n_components; ++c)
          {
            max_jump[c] += max_grad_jump[c];
          }
      }
    if (accumulate_value_jump)
      {
        const bool is_on_boundary =
          (boundary_id != numbers::invalid_unsigned_int);

        std::vector<Vector<double>> W(
          n_q_points, Vector<double>(EquationComponents<dim>::n_components));
        std::vector<Vector<double>> W_neighbor(
          n_q_points, Vector<double>(EquationComponents<dim>::n_components));

        fe_v_face_this.get_function_values(current_solution, W);
        if (!is_on_boundary)
          {
            fe_v_face_neighbor.get_function_values(current_solution,
                                                   W_neighbor);
          }

        std::array<double, EquationComponents<dim>::n_components>
          max_value_jump;
        max_value_jump.fill(0.0);
        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            if (is_on_boundary)
              {
                Vector<double> boundary_values(
                  EquationComponents<dim>::n_components);
                const Point<dim> p = fe_v_face_this.quadrature_point(q);
                if (parameters->n_mms != 1)
                  {
                    parameters->boundary_conditions[boundary_id]
                      .values.vector_value(p, boundary_values);
                  }
                else
                  // MMS: compute boundary_values according to MS.
                  {
                    typename MMS<dim>::F_V sol;
                    typename MMS<dim>::F_V src;
                    typename MMS<dim>::F_T grad;
                    mms.evaluate(p, sol, grad, src, false);
                    for (unsigned int ic = 0;
                         ic < EquationComponents<dim>::n_components;
                         ++ic)
                      {
                        boundary_values[ic] = sol[ic];
                      }
                  }

                EulerEquations<dim>::compute_Wminus(
                  parameters->boundary_conditions[boundary_id].kind,
                  fe_v_face_this.normal_vector(q),
                  W[q],
                  boundary_values,
                  W_neighbor[q]);
              }

            for (unsigned int c = 0; c < EquationComponents<dim>::n_components;
                 ++c)
              {
                max_value_jump[c] =
                  std::max(max_value_jump[c],
                           std::abs(W[q][c] - W_neighbor[q][c]));
              }
          }
        for (unsigned int c = 0; c < EquationComponents<dim>::n_components; ++c)
          {
            max_jump[c] += max_value_jump[c];
          }
      } // End if (accumulate_value_jump)

    double all_jumps = 0.0;
    for (unsigned int c = 0; c < EquationComponents<dim>::n_components; ++c)
      {
        all_jumps += max_jump[c];
      }

    return (all_jumps);
  }

#include "NSolver.inst"
} // namespace NSFEMSolver
