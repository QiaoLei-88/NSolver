//
//  NSEquation.h
//  NSolver
//
//  Created by 乔磊 on 15/3/1.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//

#ifndef __NSolver__NSEquation__
#define __NSolver__NSEquation__

#include <deal.II/base/std_cxx11/array.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/point.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/data_component_interpretation.h>



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

// Header file for MPI parallel vectors
#include <deal.II/lac/generic_linear_algebra.h>
#define USE_TRILINOS_LA
namespace LA
{
#ifdef USE_PETSC_LA
  using namespace dealii::LinearAlgebraPETSc;
#else
  using namespace dealii::LinearAlgebraTrilinos;
#endif
}

// And this again is C++:
#include <cmath>
#include <iostream>
#include <vector>
#include <string>

namespace NSFEMSolver
{
  using namespace dealii;

#ifndef __NSVector__DEFINED__
  typedef LA::MPI::Vector NSVector;
#define __NSVector__DEFINED__
#endif

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
  struct EulerEquations
  {
    // @sect4{Component description}

    // First a few variables that describe the various components of our
    // solution vector in a generic way. This includes the number of
    // components in the system (Euler's equations have one entry for momenta
    // in each spatial direction, plus the energy and density components, for
    // a total of <code>dim+2</code> components), as well as functions that
    // describe the index within the solution vector of the first momentum
    // component, the density component, and the energy density
    // component. Note that all these %numbers depend on the space dimension;
    // defining them in a generic way (rather than by implicit convention)
    // makes our code more flexible and makes it easier to later extend it,
    // for example by adding more components to the equations.
    static const unsigned int n_components             = dim + 2;
    static const unsigned int first_momentum_component = 0;
    static const unsigned int first_velocity_component = 0;
    static const unsigned int density_component        = dim;
    static const unsigned int energy_component         = dim+1;
    static const unsigned int pressure_component       = dim+1;


    enum NumericalFluxType
    {
      LaxFriedrichs,
      Roe
    };

    // When generating graphical output way down in this program, we need to
    // specify the names of the solution variables as well as how the various
    // components group into vector and scalar fields. We could describe this
    // there, but in order to keep things that have to do with the Euler
    // equation localized here and the rest of the program as generic as
    // possible, we provide this sort of information in the following two
    // functions:
    static
    std::vector<std::string>
    component_names();

    static
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    component_interpretation();

    // @sect4{Transformations between variables}

    // Next, we define the gas constant. We will set it to 1.4 in its
    // definition immediately following the declaration of this class (unlike
    // integer variables, like the ones above, static const floating point
    // member variables cannot be initialized within the class declaration in
    // C++). This value of 1.4 is representative of a gas that consists of
    // molecules composed of two atoms, such as air which consists up to small
    // traces almost entirely of $N_2$ and $O_2$.
    static double gas_gamma;

    // In the following, we will need to compute the kinetic energy and the
    // pressure from a vector of conserved variables. This we can do based on
    // the energy density and the kinetic energy $\frac 12 \rho |\mathbf v|^2
    // = \frac{|\rho \mathbf v|^2}{2\rho}$ (note that the independent
    // variables contain the momentum components $\rho v_i$, not the
    // velocities $v_i$).
    template <typename InputVector>
    static
    typename InputVector::value_type
    compute_kinetic_energy (const InputVector &W);


    template <typename InputVector>
    static
    typename InputVector::value_type
    compute_pressure (const InputVector &W);


    // Compute total energy density := pressure/(\gamma-1) + 0.5 * \rho * |v|^2
    template <typename InputVector>
    static
    typename InputVector::value_type
    compute_energy_density (const InputVector &W);


    template <typename InputVector>
    static
    typename InputVector::value_type
    compute_velocity_magnitude (const InputVector &W);


    template <typename InputVector>
    static
    void
    compute_conservative_vector (const InputVector &W,
                                 std_cxx11::array
                                 <typename InputVector::value_type, n_components>
                                 &conservative_vector);


    template <typename InputVector>
    static
    typename InputVector::value_type
    compute_temperature (const InputVector &W);


    template <typename InputVector>
    static
    typename InputVector::value_type
    compute_molecular_viscosity (const InputVector &W);


    template <typename InputVector>
    static
    typename InputVector::value_type
    compute_sound_speed (const InputVector &W);


    // Calculate entropy according to
    // @f{eqnarray*}
    // S(p,\rho)=\frac{\rho}{\gamma -1}log(\frac{p}{\rho ^ \gamma})
    // @f}
    template <typename InputVector>
    static
    typename InputVector::value_type
    compute_entropy (const InputVector &W);


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
    static
    void compute_inviscid_flux (const InputVector &W,
                                std_cxx11::array <std_cxx11::array
                                <typename InputVector::value_type, dim>,
                                n_components > &flux);

    // Compute viscos flux as if viscosity coefficent is 1. Viscosity coefficent
    // is a linear factor of viscos flux, so you can scale the viscos flux before
    // add it to system matrix. If you want to bring sensitivity of viscosity
    // coefficent into Newton matrix, you can declare the viscosity coefficient
    // as a Sacado::FAD:DFAD<> type, otherwise you can declare the viscosity
    // as a regular double type.
    template <typename InputVector, typename InputMatrix>
    static
    void compute_viscous_flux (const InputVector &W,
                               const InputMatrix &grad_w,
                               std_cxx11::array <std_cxx11::array
                               <typename InputVector::value_type, dim>,
                               n_components > &flux);

    // @sect4{EulerEquations::compute_normal_flux}

    // On the boundaries of the domain and across hanging nodes we use a
    // numerical flux function to enforce boundary conditions.  This routine
    // is the basic Lax-Friedrich's flux with a stabilization parameter
    // $\alpha$. It's form has also been given already in the introduction:
    template <typename InputVector>
    static
    void numerical_normal_flux (const Point<dim>                   &normal,
                                const InputVector                  &Wplus,
                                const InputVector                  &Wminus,
                                const double                        alpha,
                                std_cxx11::array < typename InputVector::value_type,
                                n_components> &normal_flux,
                                NumericalFluxType const &flux_type);

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
    static
    void compute_forcing_vector (const InputVector &W,
                                 std_cxx11::array < typename InputVector::value_type,
                                 n_components> &forcing,
                                 const double gravity);

    // @sect4{Dealing with boundary conditions}

    // Another thing we have to deal with is boundary conditions. To this end,
    // let us first define the kinds of boundary conditions we currently know
    // how to deal with:
    enum BoundaryKind
    {
      inflow_boundary,
      outflow_boundary,
      no_penetration_boundary,
      pressure_boundary,
      Riemann_boundary,
      MMS_BC
    };


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
    //
    // There is a little snag that makes this function unpleasant from a C++
    // language viewpoint: The output vector <code>Wminus</code> will of
    // course be modified, so it shouldn't be a <code>const</code>
    // argument. Yet it is in the implementation below, and needs to be in
    // order to allow the code to compile. The reason is that we call this
    // function at a place where <code>Wminus</code> is of type
    // <code>Table@<2,Sacado::Fad::DFad@<double@> @></code>, this being 2d
    // table with indices representing the quadrature point and the vector
    // component, respectively. We call this function with
    // <code>Wminus[q]</code> as last argument; subscripting a 2d table yields
    // a temporary accessor object representing a 1d vector, just what we want
    // here. The problem is that a temporary accessor object can't be bound to
    // a non-const reference argument of a function, as we would like here,
    // according to the C++ 1998 and 2003 standards (something that will be
    // fixed with the next standard in the form of rvalue references).  We get
    // away with making the output argument here a constant because it is the
    // <i>accessor</i> object that's constant, not the table it points to:
    // that one can still be written to. The hack is unpleasant nevertheless
    // because it restricts the kind of data types that may be used as
    // template argument to this function: a regular vector isn't going to do
    // because that one can not be written to when marked
    // <code>const</code>. With no good solution around at the moment, we'll
    // go with the pragmatic, even if not pretty, solution shown here:
    template <typename DataVector>
    static
    void
    compute_Wminus (const BoundaryKind (&boundary_kind)[n_components],
                    const Point<dim>     &normal_vector,
                    const DataVector     &Wplus,
                    const Vector<double> &boundary_values,
                    const DataVector     &Wminus);

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
    static
    void
    compute_refinement_indicators (const DoFHandler<dim> &dof_handler,
                                   const Mapping<dim>    &mapping,
                                   const NSVector &solution,
                                   Vector<double>        &refinement_indicators);
  };

  // Put the definition of nested class mememer function templates in header
  // file.

  // In the following, we will need to compute the kinetic energy and the
  // pressure from a vector of conserved variables. This we can do based on
  // the energy density and the kinetic energy $\frac 12 \rho |\mathbf v|^2
  // = \frac{|\rho \mathbf v|^2}{2\rho}$ (note that the independent
  // variables contain the momentum components $\rho v_i$, not the
  // velocities $v_i$).

  template <int dim>
  template <typename InputVector>
  typename InputVector::value_type
  EulerEquations<dim>::compute_kinetic_energy (const InputVector &W)
  {
    typename InputVector::value_type kinetic_energy = 0;
    for (unsigned int d=0; d<dim; ++d)
      kinetic_energy += W[first_velocity_component+d] *
                        W[first_velocity_component+d];
    kinetic_energy *= 0.5 * W[density_component];

    return kinetic_energy;
  }

  template <int dim>
  template <typename InputVector>
  typename InputVector::value_type
  EulerEquations<dim>::compute_pressure (const InputVector &W)
  {
    return (* (W.begin() + pressure_component));
  }

  template <int dim>
  template <typename InputVector>
  typename InputVector::value_type
  EulerEquations<dim>::compute_energy_density (const InputVector &W)
  {
    return (* (W.begin() + pressure_component)/ (gas_gamma-1.0)
            + compute_kinetic_energy (W)
           );
  }

  template <int dim>
  template <typename InputVector>
  typename InputVector::value_type
  EulerEquations<dim>::compute_velocity_magnitude (const InputVector &W)
  {
    typename InputVector::value_type velocity_magnitude = 0;
    for (unsigned int d=0; d<dim; ++d)
      velocity_magnitude += W[first_velocity_component+d] *
                            W[first_velocity_component+d];
    velocity_magnitude = std::sqrt (velocity_magnitude);

    return velocity_magnitude;
  }

  template <int dim>
  template <typename InputVector>
  void
  EulerEquations<dim>::compute_conservative_vector (const InputVector &W,
                                                    std_cxx11::array
                                                    <typename InputVector::value_type, n_components>
                                                    &conservative_vector)
  {
    for (unsigned int d = 0; d<dim; ++d)
      {
        conservative_vector[first_velocity_component+d]
          =W[first_velocity_component+d] * W[density_component];
      }
    conservative_vector[density_component] = W[density_component];
    conservative_vector[energy_component] = compute_energy_density (W);
    return;
  }

  template <int dim>
  template <typename InputVector>
  typename InputVector::value_type
  EulerEquations<dim>::compute_temperature (const InputVector &W)
  {
    return (gas_gamma * W[pressure_component]/W[density_component]);
  }


  template <int dim>
  template <typename InputVector>
  typename InputVector::value_type
  EulerEquations<dim>::compute_molecular_viscosity (const InputVector &W)
  {
    const double sutherland_const=110.4/273.15;
    const typename InputVector::value_type t=compute_temperature (W);
    return (std::pow (t,1.5) * (1.0+sutherland_const)/ (t+sutherland_const));
  }


  template <int dim>
  template <typename InputVector>
  typename InputVector::value_type
  EulerEquations<dim>::compute_sound_speed (const InputVector &W)
  {
    return (std::sqrt (compute_temperature (W)));
  }

  // Calculate entropy according to
  // @f{eqnarray*}
  // S(p,\rho)=\frac{\rho}{\gamma -1}log(\frac{p}{\rho ^ \gamma})
  // @f}
  template <int dim>
  template <typename InputVector>
  typename InputVector::value_type
  EulerEquations<dim>::compute_entropy (const InputVector &W)
  {
    return (W[density_component] / (gas_gamma-1.0) *
            std::log (W[pressure_component] /
                      std::pow (W[density_component], gas_gamma)
                     )
           );
  }


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
  template <int dim>
  template <typename InputVector>
  void EulerEquations<dim>::compute_inviscid_flux (const InputVector &W,
                                                   std_cxx11::array <std_cxx11::array
                                                   <typename InputVector::value_type, dim>,
                                                   EulerEquations<dim>::n_components > &flux)
  {
    const typename InputVector::value_type pressure = W[pressure_component];

    for (unsigned int d=0; d<dim; ++d)
      {
        for (unsigned int e=0; e<dim; ++e)
          flux[first_momentum_component+d][e]
            = W[first_velocity_component+d] *
              W[first_velocity_component+e] *
              W[density_component];

        flux[first_momentum_component+d][d] += pressure;
      }

    for (unsigned int d=0; d<dim; ++d)
      {
        flux[density_component][d]
          = W[first_velocity_component+d] * W[density_component];
      }

    for (unsigned int d=0; d<dim; ++d)
      flux[energy_component][d] = W[first_velocity_component+d] *
                                  (compute_energy_density (W) + pressure);

  }

  // Compute viscos flux as if viscosity coefficent is 1. Viscosity coefficent
  // is a linear factor of viscos flux, so you can scale the viscos flux before
  // add it to system matrix. If you want to bring sensitivity of viscosity
  // coefficent into Newton matrix, you can declare the viscosity coefficient
  // as a Sacado::FAD:DFAD<> type, otherwise you can declare the viscosity
  // as a regular double type.
  template <int dim>
  template <typename InputVector, typename InputMatrix>
  void EulerEquations<dim>::compute_viscous_flux (const InputVector &W,
                                                  const InputMatrix &grad_w,
                                                  std_cxx11::array <std_cxx11::array
                                                  <typename InputVector::value_type, dim>,
                                                  EulerEquations<dim>::n_components > &flux)
  {
    // First evaluate viscous flux's contribution to momentum equations.

    // At the first of first we evaluate shear stress tensor
    std_cxx11::array <std_cxx11::array <typename InputVector::value_type, dim>, dim>
    stress_tensor;
    typename InputVector::value_type bolk_stress = 0;
    for (unsigned int k=0; k<dim; ++k)
      {
        bolk_stress += 2.0/3.0 * grad_w[first_velocity_component+k][k];
      }

    for (unsigned int i=0; i<dim; ++i)
      {
        for (unsigned int j=0; j<=i; ++j)
          {
            stress_tensor[i][j] = (grad_w[first_velocity_component+i][j] +
                                   grad_w[first_velocity_component+j][i]);
            if (j != i)
              {
                stress_tensor[j][i] = stress_tensor[i][j];
              }
          }
        stress_tensor[i][i] -= bolk_stress;
      }
    // Submit viscosity stress to momentum equations.
    for (unsigned int d=0; d<dim; ++d)
      {
        for (unsigned int e=0; e<dim; ++e)
          {
            flux[first_momentum_component+d][e] = stress_tensor[d][e];
          }
      }

    // Viscous flux has nothing to do with mass equation, at least for now.
    for (unsigned int d=0; d<dim; ++d)
      {
        flux[density_component][d] = 0.0;
      }

    // At last deal with energy equation
    const double prandtlNumber = 0.72;
    const double heat_conductivity = 1.0 / (prandtlNumber * (gas_gamma - 1.0));
    const typename InputVector::value_type rho_inverse = 1.0/W[density_component];
    const typename InputVector::value_type p_over_rho_square =
      W[pressure_component]*rho_inverse*rho_inverse;
    for (unsigned int d=0; d<dim; ++d)
      {
        flux[energy_component][d] = 0.0;
      }
    for (unsigned int d=0; d<dim; ++d)
      {
        for (unsigned int e=0; e<dim; ++e)
          {
            flux[energy_component][d] += W[first_velocity_component+e]*stress_tensor[d][e];
          }
        // Calulate gradient of temperature. Notice that T=gamma*p/rho, then wo do
        // d_T = gamma / rho * d_p - gamma * p/(rho^2) * d_rho on every space dimension.
        flux[energy_component][d] += heat_conductivity * gas_gamma *
                                     rho_inverse * grad_w[pressure_component][d];
        flux[energy_component][d] -= heat_conductivity * gas_gamma *
                                     p_over_rho_square * grad_w[density_component][d];
      }

  }


  // @sect4{EulerEquations::compute_normal_flux}

  // On the boundaries of the domain and across hanging nodes we use a
  // numerical flux function to enforce boundary conditions.  This routine
  // is the basic Lax-Friedrich's flux with a stabilization parameter
  // $\alpha$. It's form has also been given already in the introduction:
  template <int dim>
  template <typename InputVector>
  void EulerEquations<dim>::numerical_normal_flux (const Point<dim>                   &normal,
                                                   const InputVector                  &Wplus,
                                                   const InputVector                  &Wminus,
                                                   const double                        alpha,
                                                   std_cxx11::array < typename InputVector::value_type, n_components> &normal_flux,
                                                   NumericalFluxType const &flux_type)
  {
    typedef typename InputVector::value_type VType;

    std_cxx11::array <std_cxx11::array <VType, dim>, n_components > iflux, oflux;
    compute_inviscid_flux (Wplus, iflux);
    compute_inviscid_flux (Wminus, oflux);

    switch (flux_type)
      {
      case LaxFriedrichs:
      {
        for (unsigned int ic=0; ic<n_components; ++ic)
          {
            normal_flux[ic] = 0.0;
            for (unsigned int d=0; d<dim; ++d)
              {
                normal_flux[ic] += (iflux[ic][d] + oflux[ic][d]) * normal[d];
              }
            normal_flux[ic] += alpha * (Wplus[ic] - Wminus[ic]);
            normal_flux[ic] *= 0.5;
          }
        break;
      }
      case Roe:
      {
        VType const Roe_factor = std::sqrt (Wminus[density_component]/Wplus[density_component]);
        VType const Roe_factor_plus_1 = Roe_factor + 1.0;
        // Values in Roe_average is density, velosity[1,..,dim], specified total
        // enthalpy.
        std::array<VType, n_components> Roe_average;

        VType velocity_averaged_square = 0.0;
        VType velocity_averaged_dot_n = 0.0;
        VType v_dot_n_l = 0.0;
        VType v_dot_n_r = 0.0;
        for (unsigned int ic = first_velocity_component;
             ic < first_velocity_component + dim; ++ic)
          {
            Roe_average[ic] = (Wplus[ic] + Wminus[ic] * Roe_factor)/Roe_factor_plus_1;
            velocity_averaged_square += Roe_average[ic] * Roe_average[ic];
            velocity_averaged_dot_n += Roe_average[ic] * normal[ic];
            v_dot_n_l += Wplus[ic] * normal[ic];
            v_dot_n_r += Wminus[ic] * normal[ic];
          }
        Roe_average[density_component] = Roe_factor * Wplus[density_component];

        double const gg = gas_gamma/ (gas_gamma-1);

        VType const enthalpy_l = (gg*Wplus[pressure_component] +
                                  compute_kinetic_energy (Wplus))/Wplus[density_component];
        VType const enthalpy_r = (gg*Wminus[pressure_component] +
                                  compute_kinetic_energy (Wminus))/Wminus[density_component];
        Roe_average[pressure_component] = (enthalpy_l + enthalpy_r * Roe_factor)/Roe_factor_plus_1;

        VType const sound_speed_averaged_quare = (gas_gamma-1) *
                                                 (Roe_average[pressure_component] - 0.5 * velocity_averaged_square);
        VType const sound_speed_averaged = std::sqrt (sound_speed_averaged_quare);

        // Factor for Harten's entropy correction
        VType const delta = 0.1 * sound_speed_averaged;
        std::array<VType, n_components> solution_jump;
        for (unsigned int ic=0; ic<n_components; ++ic)
          {
            solution_jump[ic] = Wminus[ic] - Wplus[ic];
          }
        VType const normal_velocity_jump = v_dot_n_r - v_dot_n_l;

        VType c1 = std::abs (velocity_averaged_dot_n - sound_speed_averaged);
        if (c1 < delta)
          {
            c1 = 0.5 * (c1*c1/delta + delta);
          }
        c1 *= (solution_jump[pressure_component] - Roe_average[density_component]
               * sound_speed_averaged * normal_velocity_jump) /
              (2.0 * sound_speed_averaged_quare);

        VType c2 = std::abs (velocity_averaged_dot_n);
        VType const delta_v = 0.2 * delta;
        if (c2 < 0.2*delta)
          {
            c2 = 0.5 * (c2*c2/delta_v + delta_v);
          }

        VType const c3 = c2 * Roe_average[density_component];
        c2 *= (solution_jump[density_component] - solution_jump[pressure_component]
               /sound_speed_averaged_quare);

        VType c4 = std::abs (velocity_averaged_dot_n + sound_speed_averaged);
        if (c4 < delta)
          {
            c4 = 0.5 * (c4*c4/delta + delta);
          }
        c4 *= (solution_jump[pressure_component] + Roe_average[density_component]
               * sound_speed_averaged * normal_velocity_jump) /
              (2.0 * sound_speed_averaged_quare);

        // Assemble Roe jump
        std::array<VType, n_components> Roe_jump;

        for (unsigned int ic = first_velocity_component, id=0; id < dim; ++ic, ++id)
          {
            Roe_jump[ic]  = c1 * (Roe_average[ic] - sound_speed_averaged * normal[id]);
            Roe_jump[ic] += c2 * Roe_average[ic];
            Roe_jump[ic] += c3 * (solution_jump[ic] - normal_velocity_jump * normal[id]);
            Roe_jump[ic] += c4 * (Roe_average[ic] + sound_speed_averaged * normal[id]);
          }
        Roe_jump[density_component] = c1 + c2 + c4;
        {
          unsigned int const ic = pressure_component;
          Roe_jump[ic]  = c1 * (Roe_average[ic] - sound_speed_averaged * velocity_averaged_dot_n);
          Roe_jump[ic] += c2 * 0.5 * velocity_averaged_square;
          VType uu = -velocity_averaged_dot_n * normal_velocity_jump;
          for (unsigned int iv = first_velocity_component;
               iv < first_velocity_component + dim; ++iv)
            {
              uu += Roe_average[iv] * solution_jump[iv];
            }
          Roe_jump[ic] += c3 * uu;
          Roe_jump[ic] += c4 * (Roe_average[ic] + sound_speed_averaged * velocity_averaged_dot_n);
        }

        // Finally, the Roe flux
        for (unsigned int ic=0; ic<n_components; ++ic)
          {
            normal_flux[ic] = 0.0;
            for (unsigned int d=0; d<dim; ++d)
              {
                normal_flux[ic] += (iflux[ic][d] + oflux[ic][d]) * normal[d];
              }
            normal_flux[ic] -= Roe_jump[ic];
            normal_flux[ic] *= 0.5;
          }
        break;
      }
      default:
        Assert (false, ExcNotImplemented());
        break;
      }


  }

  // @sect4{EulerEquations::compute_forcing_vector}

  // In the same way as describing the flux function $\mathbf F(\mathbf w)$,
  // we also need to have a way to describe the right hand side forcing
  // term. As mentioned in the introduction, we consider only gravity here,
  // which leads to the specific form $\mathbf G(\mathbf w) = \left(
  // g_1\rho, g_2\rho, g_3\rho, 0, \rho \mathbf g \cdot \mathbf v
  // \right)^T$, shown here for the 3d case. More specifically, we will
  // consider only $\mathbf g=(0,0,-1)^T$ in 3d, or $\mathbf g=(0,-1)^T$ in
  // 2d. This naturally leads to the following function:
  template <int dim>
  template <typename InputVector>
  void EulerEquations<dim>::compute_forcing_vector (const InputVector &W,
                                                    std_cxx11::array
                                                    < typename InputVector::value_type, n_components>
                                                    &forcing,
                                                    const double gravity)
  {
    for (unsigned int c=0; c<n_components; ++c)
      switch (c)
        {
        case first_momentum_component+dim-1:
          forcing[c] = gravity * W[density_component];
          break;
        case energy_component:
          forcing[c] = gravity *
                       W[density_component] *
                       W[first_velocity_component+dim-1];
          break;
        default:
          forcing[c] = 0;
          break;
        }
  }


  // @sect4{Dealing with boundary conditions}

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
  //
  // There is a little snag that makes this function unpleasant from a C++
  // language viewpoint: The output vector <code>Wminus</code> will of
  // course be modified, so it shouldn't be a <code>const</code>
  // argument. Yet it is in the implementation below, and needs to be in
  // order to allow the code to compile. The reason is that we call this
  // function at a place where <code>Wminus</code> is of type
  // <code>Table@<2,Sacado::Fad::DFad@<double@> @></code>, this being 2d
  // table with indices representing the quadrature point and the vector
  // component, respectively. We call this function with
  // <code>Wminus[q]</code> as last argument; subscripting a 2d table yields
  // a temporary accessor object representing a 1d vector, just what we want
  // here. The problem is that a temporary accessor object can't be bound to
  // a non-const reference argument of a function, as we would like here,
  // according to the C++ 1998 and 2003 standards (something that will be
  // fixed with the next standard in the form of rvalue references).  We get
  // away with making the output argument here a constant because it is the
  // <i>accessor</i> object that's constant, not the table it points to:
  // that one can still be written to. The hack is unpleasant nevertheless
  // because it restricts the kind of data types that may be used as
  // template argument to this function: a regular vector isn't going to do
  // because that one can not be written to when marked
  // <code>const</code>. With no good solution around at the moment, we'll
  // go with the pragmatic, even if not pretty, solution shown here:
  template <int dim>
  template <typename DataVector>
  void
  EulerEquations<dim>::compute_Wminus (const BoundaryKind (&boundary_kind)[n_components],
                                       const Point<dim>     &normal_vector,
                                       const DataVector     &Wplus,
                                       const Vector<double> &boundary_values,
                                       const DataVector     &Wminus)
  {
    typedef typename DataVector::value_type VType;

    for (unsigned int c = 0; c < n_components; c++)
      switch (boundary_kind[c])
        {
        case Riemann_boundary:
        {
          // Riemann boundary condition is a characteristics based boundary condtion.
          // Riemann boundary condition set values of components once for all.
          if (c == 0)
            {
              // Compute sound speed and normal velocity from demanded boundary values
              VType const sound_speed_incoming = compute_sound_speed (boundary_values);

              VType normal_velocity_incoming = 0.0;
              for (unsigned int d = first_velocity_component; d < first_velocity_component+dim; ++d)
                {
                  normal_velocity_incoming += boundary_values[d]*normal_vector[d];
                }

              if (normal_velocity_incoming + sound_speed_incoming <= 0.0)
                {
                  // This is a supersonic inflow boundary, enforce all boundary values
                  // without calculating characteristics values
                  for (unsigned int ic=0; ic < n_components; ++ic)
                    {
                      Wminus[ic] = boundary_values[ic];
                    }
                }
              else if (normal_velocity_incoming - sound_speed_incoming >= 0.0)
                {
                  // This is a supersonic outflow boundary, extrapolate all boundary values
                  // without calculating characteristics values
                  for (unsigned int ic=0; ic < n_components; ++ic)
                    {
                      Wminus[ic] = Wplus[ic];
                    }
                }
              else
                {
                  // This is a subsonic boundary. Evaulate interior normal speed and sound speed.
                  VType const sound_speed_outcoming = compute_sound_speed (Wplus);

                  VType normal_velocity_outcoming = 0.0;
                  for (unsigned int d = first_velocity_component; d < first_velocity_component+dim; ++d)
                    {
                      normal_velocity_outcoming += Wplus[d]*normal_vector[d];
                    }
                  // v_n+c > 0, thus compute v_n + 2*c/(gas_gamma - 1) from interior values.
                  VType riemann_invariant_outcoming = normal_velocity_outcoming
                                                      + 2.0 * sound_speed_outcoming/ (gas_gamma - 1.0);
                  // v_n-c < 0, thus compute v_n - 2*c/(gas_gamma - 1) from demanded boundary values.
                  VType riemann_invariant_incoming = normal_velocity_incoming
                                                     - 2.0 * sound_speed_incoming/ (gas_gamma - 1.0);

                  VType entropy_boundary;
                  std_cxx11::array<VType,dim> tangential_velocity;
                  if (normal_velocity_incoming <= 0.0)
                    {
                      // v_n <= 0
                      // This is a subsonic inflow boundary.
                      // Compute tangential velocity and entropy(not exactly) from demanded boundary values.
                      for (unsigned int d = first_velocity_component, i=0; i < dim; ++d, ++i)
                        {
                          tangential_velocity[i] = boundary_values [d] - normal_velocity_incoming * normal_vector[d];
                        }
                      entropy_boundary = std::pow (boundary_values[density_component], gas_gamma)/
                                         boundary_values[pressure_component];
                    }
                  else
                    {
                      // v_n > 0
                      // This is a subsonic outflow boundary.
                      // Compute tangential velocity and entropy(not exactly) from interior values.
                      for (unsigned int d = first_velocity_component, i=0; i < dim; ++d, ++i)
                        {
                          tangential_velocity[i] = Wplus[d] - normal_velocity_outcoming * normal_vector[d];
                        }
                      entropy_boundary = std::pow (Wplus[density_component], gas_gamma)/
                                         Wplus[pressure_component];
                    }
                  // Recover primitive variables from characteristics variables.
                  VType const normal_velocity_boundary =
                    0.5 * (riemann_invariant_outcoming + riemann_invariant_incoming);
                  VType const sound_speed_boundary =
                    0.25 * (gas_gamma - 1.0) * (riemann_invariant_outcoming - riemann_invariant_incoming);
                  Assert (sound_speed_boundary > 0.0,
                          ExcLowerRangeType<VType> (sound_speed_boundary, 0.0));

                  for (unsigned int d = first_velocity_component, i=0; i < dim; ++d, ++i)
                    {
                      Wminus[d] = tangential_velocity[i] + normal_velocity_boundary * normal_vector[d];
                    }
                  Wminus[density_component] = std::pow (sound_speed_boundary * sound_speed_boundary
                                                        * entropy_boundary / gas_gamma, 1.0/ (gas_gamma-1.0));
                  Wminus[pressure_component] = Wminus[density_component] * sound_speed_boundary *
                                               sound_speed_boundary / gas_gamma;
                }
            }
          break;
        }

        case MMS_BC:
        {
          // MMS_BC boundary condition set values of components once for all.
          // It is similar to Riemann boundary conditon, but enforce all boundary
          // values in subsonic case. The reason is the solution is manufactured
          // and the in enforced, the solution has no degree of freedom. However,
          // on supersonic out flow boundary, the solution still has to be extrapolated.
          // This is because in supersonic case the solution depends only on
          // inflow boundary.
          if (c == 0)
            {
              // Compute sound speed and normal velocity from demanded boundary values
              VType const sound_speed_incoming = compute_sound_speed (boundary_values);
              VType normal_velocity_incoming = 0.0;
              for (unsigned int d = first_velocity_component; d < first_velocity_component+dim; ++d)
                {
                  normal_velocity_incoming += boundary_values[d]*normal_vector[d];
                }

              if (normal_velocity_incoming - sound_speed_incoming >= 0.0)
                {
                  // This is a supersonic outflow boundary, extrapolate all boundary values
                  // without calculating characteristics values
                  for (unsigned int ic=0; ic < n_components; ++ic)
                    {
                      Wminus[ic] = Wplus[ic];
                    }
                }
              else
                {
                  for (unsigned int ic=0; ic < n_components; ++ic)
                    {
                      Wminus[ic] = boundary_values[ic];
                    }
                }
            }
          break;
        }

        case inflow_boundary:
        {
          Wminus[c] = boundary_values (c);
          break;
        }

        case outflow_boundary:
        {
          Wminus[c] = Wplus[c];
          break;
        }

        case pressure_boundary:
        {
          Wminus[c] = boundary_values (c);
          break;
        }

        case no_penetration_boundary:
        {
          Assert (c!=density_component,
                  ExcMessage ("Can not apply no_penetration_boundary to density."));
          Assert (c!=energy_component ,
                  ExcMessage ("Can not apply no_penetration_boundary to energy."));
          // We prescribe the velocity (we are dealing with a particular
          // component here so that the average of the velocities is
          // orthogonal to the surface normal.  This creates sensitivities of
          // across the velocity components.
          typename DataVector::value_type vdotn = 0;
          for (unsigned int d = 0; d < dim; d++)
            {
              vdotn += Wplus[d]*normal_vector[d];
            }

          Wminus[c] = Wplus[c] - 2.0*vdotn*normal_vector[c];
          break;
        }

        default:
          Assert (false, ExcNotImplemented());
          break;
        }
  }
} /* End of namespace NSFEMSolver */


#endif /* defined(__NSolver__NSEquation__) */
