//
//  NSEquation.h
//  NSolver
//
//  Created by 乔磊 on 15/3/1.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//

#ifndef __NSolver__NSEquation__
#define __NSolver__NSEquation__

#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/lac/vector.h>



// Sacado is the automatic differentiation package within Trilinos, which is
// used to find the Jacobian for a fully implicit Newton iteration:
// It is known that Trilinos::Sacado(at least until version 11.10.2) package
// will trigger a warning.
// Since we are not responsible for this, just suppress the warning by
// the following macro directive.
DEAL_II_DISABLE_EXTRA_DIAGNOSTICS
#include <Sacado.hpp>
// Recover diagnostic checks for the rest of code.
DEAL_II_ENABLE_EXTRA_DIAGNOSTICS

#include <NSolver/BoundaryType.h>
#include <NSolver/EquationComponents.h>
#include <NSolver/NumericalFlux.h>
#include <NSolver/Parameters/AllParameters.h>
#include <NSolver/types.h>


// And this again is C++:
#include <array>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace NSFEMSolver
{
  using namespace dealii;

  // @sect3{Euler equation specifics}

  // Here we define the flux function for this particular system of
  // conservation laws, as well as pretty much everything else that's specific
  // to the Euler equations for gas dynamics, for reasons discussed in the
  // introduction. We group all this into a structure that defines everything
  // that has to do with the flux. All members of this structure are static,
  // i.e. the structure has no actual state specified by instance member
  // variables. The better way to do this, rather than a structure with all
  // static members would be to use a namespace -- but namespaces can't be
  // templatized and we want some of the member variables of the structure to
  // depend on the space dimension, which we in our usual way introduce using
  // a template parameter.
  template <int dim>
  class EulerEquations
  {
  public:
    // @sect4{Transformations between variables}

    /**
     * Inform the EulerEquations class with a pointer to runtime parameter.
     * The EulerEquations class needs to know free stream conditions, diffusion
     * constant and gravity from the run time parameter.
     */
    static void
    set_parameter(const Parameters::AllParameters<dim> *const para_ptr_in);

    // In the following, we will need to compute the kinetic energy and the
    // pressure from a vector of conserved variables. This we can do based on
    // the energy density and the kinetic energy $\frac 12 \rho |\mathbf v|^2
    // = \frac{|\rho \mathbf v|^2}{2\rho}$ (note that the independent
    // variables contain the momentum components $\rho v_i$, not the
    // velocities $v_i$).
    template <typename InputVector>
    static typename InputVector::value_type
    compute_kinetic_energy(const InputVector &W);


    template <typename InputVector>
    static typename InputVector::value_type
    compute_pressure(const InputVector &W);


    // Compute total energy density := pressure/(\gamma-1) + 0.5 * \rho * |v|^2
    template <typename InputVector>
    static typename InputVector::value_type
    compute_energy_density(const InputVector &W);


    template <typename InputVector>
    static typename InputVector::value_type
    compute_velocity_magnitude(const InputVector &W);


    template <typename InputVector>
    static void
    compute_conservative_vector(
      const InputVector &                                W,
      std::array<typename InputVector::value_type,
                 EquationComponents<dim>::n_components> &conservative_vector);


    template <typename InputVector>
    static typename InputVector::value_type
    compute_temperature(const InputVector &W);


    template <typename InputVector>
    static typename InputVector::value_type
    compute_molecular_viscosity(const InputVector &W);


    template <typename InputVector>
    static typename InputVector::value_type
    compute_sound_speed(const InputVector &W);


    // Calculate entropy according to
    // @f{eqnarray*}
    // S(p,\rho)=\frac{\rho}{\gamma -1}log(\frac{p}{\rho ^ \gamma})
    // @f}
    template <typename InputVector>
    static typename InputVector::value_type
    compute_entropy(const InputVector &W);


    // @sect4{EulerEquations::compute_inviscid_flux}

    // We define the flux function $F(W)$ as one large matrix.  Each row of
    // this matrix represents a scalar conservation law for the component in
    // that row.  The exact form of this matrix is given in the
    // introduction. Note that we know the size of the matrix: it has as many
    // rows as the system has components, and <code>dim</code> columns; rather
    // than using a FullMatrix object for such a matrix (which has a variable
    // number of rows and columns and must therefore allocate memory on the
    // heap each time such a matrix is created), we use a rectangular array of
    // numbers right away.
    //
    // We templatize the numerical type of the flux function so that we may
    // use the automatic differentiation type here.  Similarly, we will call
    // the function with different input vector data types, so we templatize
    // on it as well:
    template <typename InputVector>
    static void
    compute_inviscid_flux(
      const InputVector &                                W,
      std::array<std::array<typename InputVector::value_type, dim>,
                 EquationComponents<dim>::n_components> &flux);

    // Compute viscous flux according to provided @p extra_dynamic_viscosity.
    // @p extra_thermal_conductivity. If equation is in Euler mode, no physical
    // dynamic viscosity and thermal conductivity will be added.
    // If equation is in NavierStokes mode, an exception will be triggered.
    //
    // If you want to bring sensitivity of viscosity
    // coefficient into Newton matrix, you can declare the viscosity coefficient
    // as a Sacado::FAD:DFAD<> type, otherwise you can declare the viscosity
    // as a regular double type.
    template <typename InputVector, typename InputMatrix>
    static void
    compute_viscous_flux(
      const InputVector &                                W,
      const InputMatrix &                                grad_w,
      std::array<std::array<typename InputVector::value_type, dim>,
                 EquationComponents<dim>::n_components> &flux,
      const double artificial_dynamic_viscosity,
      const double artificial_thermal_conductivity);

    // @sect4{EulerEquations::compute_normal_flux}

    // On the boundaries of the domain and across hanging nodes we use a
    // numerical flux function to enforce boundary conditions.  This routine
    // is the basic Lax-Friedrich's flux with a stabilization parameter
    // $\alpha$. It's form has also been given already in the introduction:
    template <typename InputVector>
    static void
    numerical_normal_flux(
      const Tensor<1, dim> &                             normal,
      const InputVector &                                Wplus,
      const InputVector &                                Wminus,
      const double                                       alpha,
      std::array<typename InputVector::value_type,
                 EquationComponents<dim>::n_components> &normal_flux,
      NumericalFlux::Type const &                        flux_type);

    // @sect4{EulerEquations::compute_forcing_vector}

    // In the same way as describing the flux function $\mathbf F(\mathbf w)$,
    // we also need to have a way to describe the right hand side forcing
    // term. As mentioned in the introduction, we consider only gravity here,
    // which leads to the specific form $\mathbf G(\mathbf w) = \left(
    // g_1\rho, g_2\rho, g_3\rho, 0, \rho \mathbf g \cdot \mathbf v
    // \right)^T$, shown here for the 3d case. More specifically, we will
    // consider only $\mathbf g=(0,0,-1)^T$ in 3d, or $\mathbf g=(0,-1)^T$ in
    // 2d. This naturally leads to the following function:
    template <typename InputVector>
    static void
    compute_forcing_vector(
      const InputVector &                                W,
      std::array<typename InputVector::value_type,
                 EquationComponents<dim>::n_components> &forcing);

    // @sect4{Dealing with boundary conditions}

    // Another thing we have to deal with is boundary conditions. To this end,
    // let us first define the kinds of boundary conditions we currently know
    // how to deal with:


    // The next part is to actually decide what to do at each kind of
    // boundary. To this end, remember from the introduction that boundary
    // conditions are specified by choosing a value $\mathbf w^-$ on the
    // outside of a boundary given an inhomogeneity $\mathbf j$ and possibly
    // the solution's value $\mathbf w^+$ on the inside. Both are then passed
    // to the numerical flux $\mathbf H(\mathbf{w}^+, \mathbf{w}^-,
    // \mathbf{n})$ to define boundary contributions to the bilinear form.
    //
    // Boundary conditions can in some cases be specified for each component
    // of the solution vector independently. For example, if component $c$ is
    // marked for inflow, then $w^-_c = j_c$. If it is an outflow, then $w^-_c
    // = w^+_c$. These two simple cases are handled first in the function
    // below.
    template <typename DataVector>
    static void
    compute_Wminus(const Boundary::Type &boundary_kind,
                   const Tensor<1, dim> &normal_vector,
                   const DataVector &    Wplus,
                   const Vector<double> &boundary_values,
                   DataVector &          Wminus);

    // @sect4{EulerEquations::compute_refinement_indicators}

    // In this class, we also want to specify how to refine the mesh. The
    // class <code>NSolver</code> that will use all the information we
    // provide here in the <code>EulerEquation</code> class is pretty agnostic
    // about the particular conservation law it solves: as doesn't even really
    // care how many components a solution vector has. Consequently, it can't
    // know what a reasonable refinement indicator would be. On the other
    // hand, here we do, or at least we can come up with a reasonable choice:
    // we simply look at the gradient of the density, and compute
    // $\eta_K=\log\left(1+|\nabla\rho(x_K)|\right)$, where $x_K$ is the
    // center of cell $K$.
    //
    // There are certainly a number of equally reasonable refinement
    // indicators, but this one does, and it is easy to compute:
    static void
    compute_refinement_indicators(DoFHandler<dim> const &dof_handler,
                                  Mapping<dim> const &   mapping,
                                  NSVector const &       solution,
                                  Vector<float> &        refinement_indicators,
                                  ComponentMask const &  component_mask);

  private:
    static unsigned const n_components = EquationComponents<dim>::n_components;
    static unsigned const first_momentum_component =
      EquationComponents<dim>::first_momentum_component;
    static unsigned const first_velocity_component =
      EquationComponents<dim>::first_velocity_component;
    static unsigned const density_component =
      EquationComponents<dim>::density_component;
    static unsigned const energy_component =
      EquationComponents<dim>::energy_component;
    static unsigned const pressure_component =
      EquationComponents<dim>::pressure_component;

    static Parameters::AllParameters<dim> const *parameters;
    /**
     * Gas heat capacity ratio
     */
    static double gas_gamma;
  };

#include "NSEquation.templates.h"
} /* End of namespace NSFEMSolver */


#endif /* defined(__NSolver__NSEquation__) */
