//
//  NSEquation.h
//  NSolver
//
//  Created by 乔磊 on 15/3/1.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//

#ifndef __NSolver__NSEquation__
#define __NSolver__NSEquation__

// First a standard set of deal.II includes. Nothing special to comment on
// here:
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/std_cxx11/array.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>

//Include grid_tools to scale mesh.
#include <deal.II/grid/grid_tools.h>

#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/solution_transfer.h>

// Then, as mentioned in the introduction, we use various Trilinos packages as
// linear solvers as well as for automatic differentiation. These are in the
// following include files.
//
// Since deal.II provides interfaces to the basic Trilinos matrices, vectors,
// preconditioners and solvers, we include them similarly as deal.II linear
// algebra structures.
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>


// Sacado is the automatic differentiation package within Trilinos, which is
// used to find the Jacobian for a fully implicit Newton iteration:
#include <Sacado.hpp>

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
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>

#include <vector>

namespace NSolver
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
    static const double gas_gamma;


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
    compute_sound_speed (const InputVector &W);


    // Calculate entropy according to
    // @f{eqnarray*}
    // S(p,\rho)=\frac{\rho}{\gamma -1}log(\frac{p}{\rho ^ \gamma})
    // @f}
    template <typename InputVector>
    static
    typename InputVector::value_type
    compute_entropy (const InputVector &W);


    // @sect4{EulerEquations::compute_flux_matrix}

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
    void compute_flux_matrix (const InputVector &W,
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
                                n_components> &normal_flux);

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
      Riemann_boundary
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
    // class <code>ConservationLaw</code> that will use all the information we
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
                                   const LA::MPI::Vector &solution,
                                   Vector<double>        &refinement_indicators);

    // @sect4{EulerEquations::Postprocessor}

    // Finally, we declare a class that implements a postprocessing of data
    // components. The problem this class solves is that the variables in the
    // formulation of the Euler equations we use are in conservative rather
    // than physical form: they are momentum densities $\mathbf m=\rho\mathbf
    // v$, density $\rho$, and energy density $E$. What we would like to also
    // put into our output file are velocities $\mathbf v=\frac{\mathbf
    // m}{\rho}$ and pressure $p=(\gamma-1)(E-\frac{1}{2} \rho |\mathbf
    // v|^2)$.
    //
    // In addition, we would like to add the possibility to generate schlieren
    // plots. Schlieren plots are a way to visualize shocks and other sharp
    // interfaces. The word "schlieren" is a German word that may be
    // translated as "striae" -- it may be simpler to explain it by an
    // example, however: schlieren is what you see when you, for example, pour
    // highly concentrated alcohol, or a transparent saline solution, into
    // water; the two have the same color, but they have different refractive
    // indices and so before they are fully mixed light goes through the
    // mixture along bent rays that lead to brightness variations if you look
    // at it. That's "schlieren". A similar effect happens in compressible
    // flow because the refractive index depends on the pressure (and
    // therefore the density) of the gas.
    //
    // The origin of the word refers to two-dimensional projections of a
    // three-dimensional volume (we see a 2d picture of the 3d fluid). In
    // computational fluid dynamics, we can get an idea of this effect by
    // considering what causes it: density variations. Schlieren plots are
    // therefore produced by plotting $s=|\nabla \rho|^2$; obviously, $s$ is
    // large in shocks and at other highly dynamic places. If so desired by
    // the user (by specifying this in the input file), we would like to
    // generate these schlieren plots in addition to the other derived
    // quantities listed above.
    //
    // The implementation of the algorithms to compute derived quantities from
    // the ones that solve our problem, and to output them into data file,
    // rests on the DataPostprocessor class. It has extensive documentation,
    // and other uses of the class can also be found in step-29. We therefore
    // refrain from extensive comments.
    class Postprocessor : public DataPostprocessor<dim>
    {
    public:
      Postprocessor (const bool do_schlieren_plot);

      virtual
      void
      compute_derived_quantities_vector (const std::vector<Vector<double> >              &uh,
                                         const std::vector<std::vector<Tensor<1,dim> > > &duh,
                                         const std::vector<std::vector<Tensor<2,dim> > > &dduh,
                                         const std::vector<Point<dim> >                  &normals,
                                         const std::vector<Point<dim> >                  &evaluation_points,
                                         std::vector<Vector<double> >                    &computed_quantities) const;

      virtual std::vector<std::string> get_names() const;

      virtual
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
      get_data_component_interpretation() const;

      virtual UpdateFlags get_needed_update_flags() const;

    private:
      const bool do_schlieren_plot;
    };
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


  // @sect4{EulerEquations::compute_flux_matrix}

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
  void EulerEquations<dim>::compute_flux_matrix (const InputVector &W,
                                                 std_cxx11::array <std_cxx11::array
                                                 <typename InputVector::value_type, dim>,
                                                 EulerEquations<dim>::n_components > &flux)
  {
    // First compute the pressure that appears in the flux matrix, and then
    // compute the first <code>dim</code> columns of the matrix that
    // correspond to the momentum terms:
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

    // Then the terms for the density (i.e. mass conservation), and, lastly,
    // conservation of energy:
    for (unsigned int d=0; d<dim; ++d)
      {
        flux[density_component][d]
          = W[first_velocity_component+d] * W[density_component];
      }

    for (unsigned int d=0; d<dim; ++d)
      flux[energy_component][d] = W[first_velocity_component+d] *
                                  (compute_energy_density (W) + pressure);
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
                                                   std_cxx11::array < typename InputVector::value_type, n_components> &normal_flux)
  {
    std_cxx11::array <std_cxx11::array
    <typename InputVector::value_type, dim>,
    EulerEquations<dim>::n_components > iflux, oflux;
    compute_flux_matrix (Wplus, iflux);
    compute_flux_matrix (Wminus, oflux);

    for (unsigned int di=0; di<n_components; ++di)
      {
        normal_flux[di] = 0;
        for (unsigned int d=0; d<dim; ++d)
          {
            normal_flux[di] += 0.5* (iflux[di][d] + oflux[di][d]) * normal[d];
          }

        normal_flux[di] += 0.5*alpha* (Wplus[di] - Wminus[di]);
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
    for (unsigned int c = 0; c < n_components; c++)
      switch (boundary_kind[c])
        {
        case Riemann_boundary:
        {
          //Riemann boundary condition set values of components once for all.
          if (c == 0)
            {
              //Calculate incoming Riemann invariant from far field conditions.
              //Farfield qualities are treated in nondimensinal manner, thus
              //Density == 1.0
              //Presure == 1.0/gas_gamma
              //Sound speed == 1.0
              //velocity magnitude == Mach number.
              const typename DataVector::value_type pressure_incoming = 1.0/gas_gamma;
              const typename DataVector::value_type sound_speed_incoming = 1.0;

              typename DataVector::value_type normal_velocity_incoming = 0.0;
              for (unsigned int d = first_velocity_component; d < first_velocity_component+dim; ++d)
                {
                  normal_velocity_incoming += boundary_values (d)*normal_vector[d];
                }
              typename DataVector::value_type riemann_invariant_incoming = normal_velocity_incoming
                  - 2.0 * sound_speed_incoming/ (gas_gamma - 1.0);
              const typename DataVector::value_type
              mach_incoming = std::fabs (normal_velocity_incoming) / sound_speed_incoming;

              //Calculate outcoming Riemann invariant from interior conditions
              const typename DataVector::value_type pressure_outcoming
                = Wplus[pressure_component];
              //   = compute_pressure<typename DataVector::value_type> (Wplus);
              const typename DataVector::value_type sound_speed_outcoming
                = std::sqrt (gas_gamma * pressure_outcoming / Wplus[density_component]);

              typename DataVector::value_type normal_velocity_outcoming = 0.0;
              for (unsigned int d = first_velocity_component; d < first_velocity_component+dim; ++d)
                {
                  normal_velocity_outcoming += Wplus[d]*normal_vector[d];
                }
              //normal_velocity_outcoming /= Wplus[density_component];

              typename DataVector::value_type riemann_invariant_outcoming = normal_velocity_outcoming
                  + 2.0 * sound_speed_outcoming/ (gas_gamma - 1.0);
              const typename DataVector::value_type
              mach_outcoming = std::fabs (normal_velocity_outcoming) / sound_speed_outcoming;
              //Adjust Riemann invairant according to local Mach number
              //First case: inflow boundary with a supersonic speed,  then no out going characteristic
              //wave exists and the incoming Riemann invariant dominates.
              if (normal_velocity_incoming < 0.0 && mach_incoming  >= 1.0)
                {
                  riemann_invariant_outcoming = riemann_invariant_incoming;
                }
              //Second case: outflow boundary with a supersonic speed, then no in going characteristic
              //wave exists and the outcoming Riemann invariant dominates.
              if (normal_velocity_incoming > 0.0 && mach_outcoming >= 1.0)
                {
                  riemann_invariant_incoming = riemann_invariant_outcoming;
                }
              //Calculate boundary values using Riemann invarant
              const typename DataVector::value_type normal_velocity_boundary =
                0.5 * (riemann_invariant_outcoming + riemann_invariant_incoming);
              const typename DataVector::value_type sound_speed_boundary =
                0.25 * (gas_gamma - 1.0) * (riemann_invariant_outcoming - riemann_invariant_incoming);
              //Finally, component values of boundary element
              Assert (sound_speed_boundary > 0.1, ExcLowerRangeType<typename DataVector::value_type> (sound_speed_boundary , 0.1));
              if (normal_velocity_boundary <= 0.0)
                {
                  //For inflow boundary, boundary values are determined by the far field conditions
                  Wminus[density_component] = boundary_values (density_component) *
                                              std::pow (sound_speed_boundary/sound_speed_incoming, 2.0/ (gas_gamma-1.0));
                  const typename DataVector::value_type
                  deltaV = normal_velocity_boundary - normal_velocity_incoming;
                  for (unsigned int d = first_velocity_component; d < first_velocity_component+dim; ++d)
                    {
                      Wminus[d] = boundary_values (d) + deltaV * normal_vector[d];
                    }
                }
              else
                {
                  //For outflow boundary, boundary values are calculated from inteiror value
                  Wminus[density_component] = Wplus[density_component] *
                                              std::pow (sound_speed_boundary/sound_speed_outcoming, 2.0/ (gas_gamma-1.0));
                  const typename DataVector::value_type
                  deltaV = normal_velocity_boundary - normal_velocity_outcoming;
                  for (unsigned int d = first_velocity_component; d < first_velocity_component+dim; ++d)
                    {
                      Wminus[d] = Wplus[d] + deltaV * normal_vector[d];
                    }
                }
              //Calculate boundary pressure then energy in the same pattern.
              const typename DataVector::value_type pressure_boundary = Wminus[density_component] * sound_speed_boundary *
                                                                        sound_speed_boundary / gas_gamma;
              Wminus[pressure_component] = pressure_boundary;
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
        }
  }

} /* End of namespace NSolver */


#endif /* defined(__NSolver__NSEquation__) */
