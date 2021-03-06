//
//  NSEquation.templates.h
//  NSolver
//
//  Created by 乔磊 on 15/10/24.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//

#ifndef __NSolver__NSEquation_Templates_h__
#define __NSolver__NSEquation_Templates_h__

template <int dim>
template <typename InputVector>
typename InputVector::value_type
EulerEquations<dim>::compute_kinetic_energy(const InputVector &W)
{
  typename InputVector::value_type kinetic_energy = 0;
  for (unsigned int d = 0; d < dim; ++d)
    kinetic_energy +=
      W[first_velocity_component + d] * W[first_velocity_component + d];
  kinetic_energy *= 0.5 * W[density_component];

  return kinetic_energy;
}

template <int dim>
template <typename InputVector>
typename InputVector::value_type
EulerEquations<dim>::compute_pressure(const InputVector &W)
{
  return (W[pressure_component]);
}

template <int dim>
template <typename InputVector>
typename InputVector::value_type
EulerEquations<dim>::compute_energy_density(const InputVector &W)
{
  return (W[pressure_component] / (gas_gamma - 1.0) +
          compute_kinetic_energy(W));
}

template <int dim>
template <typename InputVector>
typename InputVector::value_type
EulerEquations<dim>::compute_velocity_magnitude(const InputVector &W)
{
  typename InputVector::value_type velocity_magnitude = 0;
  for (unsigned int d = 0; d < dim; ++d)
    velocity_magnitude +=
      W[first_velocity_component + d] * W[first_velocity_component + d];
  velocity_magnitude = std::sqrt(velocity_magnitude);

  return velocity_magnitude;
}

template <int dim>
template <typename InputVector>
void
EulerEquations<dim>::compute_conservative_vector(
  const InputVector &                                W,
  std::array<typename InputVector::value_type,
             EquationComponents<dim>::n_components> &conservative_vector)
{
  for (unsigned int d = 0; d < dim; ++d)
    {
      conservative_vector[first_velocity_component + d] =
        W[first_velocity_component + d] * W[density_component];
    }
  conservative_vector[density_component] = W[density_component];
  conservative_vector[energy_component]  = compute_energy_density(W);
  return;
}

template <int dim>
template <typename InputVector>
typename InputVector::value_type
EulerEquations<dim>::compute_temperature(const InputVector &W)
{
  return (gas_gamma * W[pressure_component] / W[density_component]);
}


template <int dim>
template <typename InputVector>
typename InputVector::value_type
EulerEquations<dim>::compute_molecular_viscosity(const InputVector &W)
{
  const double                           sutherland_const = 110.4 / 273.15;
  const typename InputVector::value_type t = compute_temperature(W);
  return (std::pow(t, 1.5) * (1.0 + sutherland_const) / (t + sutherland_const));
}


template <int dim>
template <typename InputVector>
typename InputVector::value_type
EulerEquations<dim>::compute_sound_speed(const InputVector &W)
{
  return (std::sqrt(compute_temperature(W)));
}

// Calculate entropy according to
// @f{eqnarray*}
// S(p,\rho)=\frac{\rho}{\gamma -1}log(\frac{p}{\rho ^ \gamma})
// @f}
template <int dim>
template <typename InputVector>
typename InputVector::value_type
EulerEquations<dim>::compute_entropy(const InputVector &W)
{
  return (W[density_component] / (gas_gamma - 1.0) *
          std::log(W[pressure_component] /
                   std::pow(W[density_component], gas_gamma)));
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
void
EulerEquations<dim>::compute_inviscid_flux(
  const InputVector &                                W,
  std::array<std::array<typename InputVector::value_type, dim>,
             EquationComponents<dim>::n_components> &flux)
{
  const typename InputVector::value_type pressure = W[pressure_component];

  for (unsigned int d = 0; d < dim; ++d)
    {
      for (unsigned int e = 0; e < dim; ++e)
        flux[first_momentum_component + d][e] =
          W[first_velocity_component + d] * W[first_velocity_component + e] *
          W[density_component];

      flux[first_momentum_component + d][d] += pressure;
    }

  for (unsigned int d = 0; d < dim; ++d)
    {
      flux[density_component][d] =
        W[first_velocity_component + d] * W[density_component];
    }

  for (unsigned int d = 0; d < dim; ++d)
    flux[energy_component][d] =
      W[first_velocity_component + d] * (compute_energy_density(W) + pressure);
}

// Compute viscous flux as if viscosity coefficient is 1. Viscosity coefficient
// is a linear factor of viscous flux, so you can scale the viscous flux before
// add it to system matrix. If you want to bring sensitivity of viscosity
// coefficient into Newton matrix, you can declare the viscosity coefficient
// as a Sacado::FAD:DFAD<> type, otherwise you can declare the viscosity
// as a regular double type.
template <int dim>
template <typename InputVector, typename InputMatrix>
void
EulerEquations<dim>::compute_viscous_flux(
  const InputVector &                                W,
  const InputMatrix &                                grad_w,
  std::array<std::array<typename InputVector::value_type, dim>,
             EquationComponents<dim>::n_components> &flux,
  const double artificial_dynamic_viscosity,
  const double artificial_thermal_conductivity)
{
  Assert(parameters, ExcMessage("Null pointer encountered!"));
  // Set up viscous and thermal conductivity coefficients
  const double prandtlNumber              = 0.72;
  double       physical_dynamic_viscosity = 0.0;
  double       physical_thermal_conductivity =
    physical_dynamic_viscosity / (prandtlNumber * (gas_gamma - 1.0));

  if (parameters->equation_type == Parameters::PhysicalParameters::NavierStokes)
    {
      AssertThrow(false, ExcNotImplemented());
    }
  const double dynamic_viscosity =
    std::max(artificial_dynamic_viscosity, physical_dynamic_viscosity);
  const double thermal_conductivity =
    std::max(artificial_thermal_conductivity, physical_thermal_conductivity);
  const double bolk_viscosity = -2.0 * dynamic_viscosity / 3.0;

  // First evaluate viscous flux's contribution to momentum equations.
  // At the first of first we evaluate shear stress tensor
  std::array<std::array<typename InputVector::value_type, dim>, dim>
    stress_tensor;

  typename InputVector::value_type bolk_strain = 0;
  for (unsigned int k = 0; k < dim; ++k)
    {
      bolk_strain += grad_w[first_velocity_component + k][k];
    }

  for (unsigned int i = 0; i < dim; ++i)
    {
      for (unsigned int j = 0; j < i; ++j)
        {
          stress_tensor[i][j] =
            dynamic_viscosity * (grad_w[first_velocity_component + i][j] +
                                 grad_w[first_velocity_component + j][i]);
          stress_tensor[j][i] = stress_tensor[i][j];
        }
      stress_tensor[i][i] =
        dynamic_viscosity * 2.0 * grad_w[first_velocity_component + i][i] +
        bolk_viscosity * bolk_strain;
    }
  // Submit viscosity stress to momentum equations.
  for (unsigned int d = 0; d < dim; ++d)
    {
      for (unsigned int e = 0; e < dim; ++e)
        {
          flux[first_momentum_component + d][e] = stress_tensor[d][e];
        }
    }

  // Viscous flux for mass equation.
  const double mass_diffusion = dynamic_viscosity;
  for (unsigned int d = 0; d < dim; ++d)
    {
      flux[density_component][d] =
        mass_diffusion * grad_w[density_component][d];
    }

  // At last deal with energy equation.
  const typename InputVector::value_type rho_inverse =
    1.0 / W[density_component];
  const typename InputVector::value_type p_over_rho_square =
    W[pressure_component] * rho_inverse * rho_inverse;
  for (unsigned int d = 0; d < dim; ++d)
    {
      flux[energy_component][d] = 0.0;
    }
  for (unsigned int d = 0; d < dim; ++d)
    {
      for (unsigned int e = 0; e < dim; ++e)
        {
          flux[energy_component][d] +=
            W[first_velocity_component + e] * stress_tensor[d][e];
        }
      // Calculate gradient of temperature. Notice that T=gamma*p/rho, then wo
      // do d_T = gamma / rho * d_p - gamma * p/(rho^2) * d_rho on every space
      // dimension.
      flux[energy_component][d] += thermal_conductivity * gas_gamma *
                                   rho_inverse * grad_w[pressure_component][d];
      flux[energy_component][d] -= thermal_conductivity * gas_gamma *
                                   p_over_rho_square *
                                   grad_w[density_component][d];
    }

  // Scale with parameter
  for (unsigned int ic = 0; ic < n_components; ++ic)
    for (unsigned int id = 0; id < dim; ++id)
      {
        flux[ic][id] *= parameters->diffusion_factor[ic];
      }
  return;
}


// @sect4{EulerEquations::compute_normal_flux}

// On the boundaries of the domain and across hanging nodes we use a
// numerical flux function to enforce boundary conditions.  This routine
// is the basic Lax-Friedrich's flux with a stabilization parameter
// $\alpha$. It's form has also been given already in the introduction:
template <int dim>
template <typename InputVector>
void
EulerEquations<dim>::numerical_normal_flux(
  const Tensor<1, dim> &                             normal,
  const InputVector &                                Wplus,
  const InputVector &                                Wminus,
  const double                                       alpha,
  std::array<typename InputVector::value_type,
             EquationComponents<dim>::n_components> &normal_flux,
  NumericalFlux::Type const &                        flux_type)
{
  typedef typename InputVector::value_type VType;

  std::array<std::array<VType, dim>, n_components> iflux, oflux;
  compute_inviscid_flux(Wplus, iflux);
  compute_inviscid_flux(Wminus, oflux);

  switch (flux_type)
    {
      case NumericalFlux::LaxFriedrichs:
        {
          for (unsigned int ic = 0; ic < n_components; ++ic)
            {
              normal_flux[ic] = 0.0;
              for (unsigned int d = 0; d < dim; ++d)
                {
                  normal_flux[ic] += (iflux[ic][d] + oflux[ic][d]) * normal[d];
                }
              normal_flux[ic] += alpha * (Wplus[ic] - Wminus[ic]);
              normal_flux[ic] *= 0.5;
            }
          break;
        }
      case NumericalFlux::Roe:
        {
          VType const Roe_factor =
            std::sqrt(Wminus[density_component] / Wplus[density_component]);
          VType const Roe_factor_plus_1 = Roe_factor + 1.0;
          // Values in Roe_average is density, velocity[1,..,dim], specified
          // total enthalpy.
          std::array<VType, n_components> Roe_average;

          VType velocity_averaged_square = 0.0;
          VType velocity_averaged_dot_n  = 0.0;
          VType v_dot_n_l                = 0.0;
          VType v_dot_n_r                = 0.0;
          for (unsigned int ic = first_velocity_component;
               ic < first_velocity_component + dim;
               ++ic)
            {
              Roe_average[ic] =
                (Wplus[ic] + Wminus[ic] * Roe_factor) / Roe_factor_plus_1;
              velocity_averaged_square += Roe_average[ic] * Roe_average[ic];
              velocity_averaged_dot_n += Roe_average[ic] * normal[ic];
              v_dot_n_l += Wplus[ic] * normal[ic];
              v_dot_n_r += Wminus[ic] * normal[ic];
            }
          Roe_average[density_component] =
            Roe_factor * Wplus[density_component];

          double const gg = gas_gamma / (gas_gamma - 1);

          VType const enthalpy_l =
            (gg * Wplus[pressure_component] + compute_kinetic_energy(Wplus)) /
            Wplus[density_component];
          VType const enthalpy_r =
            (gg * Wminus[pressure_component] + compute_kinetic_energy(Wminus)) /
            Wminus[density_component];
          Roe_average[pressure_component] =
            (enthalpy_l + enthalpy_r * Roe_factor) / Roe_factor_plus_1;

          VType const sound_speed_averaged_quare =
            (gas_gamma - 1) *
            (Roe_average[pressure_component] - 0.5 * velocity_averaged_square);
          VType const sound_speed_averaged =
            std::sqrt(sound_speed_averaged_quare);

          // Factor for Harten's entropy correction
          VType const                     delta = 0.1 * sound_speed_averaged;
          std::array<VType, n_components> solution_jump;
          for (unsigned int ic = 0; ic < n_components; ++ic)
            {
              solution_jump[ic] = Wminus[ic] - Wplus[ic];
            }
          VType const normal_velocity_jump = v_dot_n_r - v_dot_n_l;

          VType c1 = std::abs(velocity_averaged_dot_n - sound_speed_averaged);
          if (c1 < delta)
            {
              c1 = 0.5 * (c1 * c1 / delta + delta);
            }
          c1 *= (solution_jump[pressure_component] -
                 Roe_average[density_component] * sound_speed_averaged *
                   normal_velocity_jump) /
                (2.0 * sound_speed_averaged_quare);

          VType       c2      = std::abs(velocity_averaged_dot_n);
          VType const delta_v = 0.2 * delta;
          if (c2 < 0.2 * delta)
            {
              c2 = 0.5 * (c2 * c2 / delta_v + delta_v);
            }

          VType const c3 = c2 * Roe_average[density_component];
          c2 *=
            (solution_jump[density_component] -
             solution_jump[pressure_component] / sound_speed_averaged_quare);

          VType c4 = std::abs(velocity_averaged_dot_n + sound_speed_averaged);
          if (c4 < delta)
            {
              c4 = 0.5 * (c4 * c4 / delta + delta);
            }
          c4 *= (solution_jump[pressure_component] +
                 Roe_average[density_component] * sound_speed_averaged *
                   normal_velocity_jump) /
                (2.0 * sound_speed_averaged_quare);

          // Assemble Roe jump
          std::array<VType, n_components> Roe_jump;

          for (unsigned int ic = first_velocity_component, id = 0; id < dim;
               ++ic, ++id)
            {
              Roe_jump[ic] =
                c1 * (Roe_average[ic] - sound_speed_averaged * normal[id]);
              Roe_jump[ic] += c2 * Roe_average[ic];
              Roe_jump[ic] +=
                c3 * (solution_jump[ic] - normal_velocity_jump * normal[id]);
              Roe_jump[ic] +=
                c4 * (Roe_average[ic] + sound_speed_averaged * normal[id]);
            }
          Roe_jump[density_component] = c1 + c2 + c4;
          {
            unsigned int const ic = pressure_component;
            Roe_jump[ic] = c1 * (Roe_average[ic] - sound_speed_averaged *
                                                     velocity_averaged_dot_n);
            Roe_jump[ic] += c2 * 0.5 * velocity_averaged_square;
            VType uu = -velocity_averaged_dot_n * normal_velocity_jump;
            for (unsigned int iv = first_velocity_component;
                 iv < first_velocity_component + dim;
                 ++iv)
              {
                uu += Roe_average[iv] * solution_jump[iv];
              }
            Roe_jump[ic] += c3 * uu;
            Roe_jump[ic] += c4 * (Roe_average[ic] + sound_speed_averaged *
                                                      velocity_averaged_dot_n);
          }

          // Finally, the Roe flux
          for (unsigned int ic = 0; ic < n_components; ++ic)
            {
              normal_flux[ic] = 0.0;
              for (unsigned int d = 0; d < dim; ++d)
                {
                  normal_flux[ic] += (iflux[ic][d] + oflux[ic][d]) * normal[d];
                }
              normal_flux[ic] -= Roe_jump[ic];
              normal_flux[ic] *= 0.5;
            }
          break;
        }
      default:
        Assert(false, ExcNotImplemented());
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
void
EulerEquations<dim>::compute_forcing_vector(
  const InputVector &                                W,
  std::array<typename InputVector::value_type,
             EquationComponents<dim>::n_components> &forcing)
{
  Assert(parameters, ExcMessage("Null pointer encountered!"));
  // For momentum equations
  for (unsigned int ic = first_momentum_component, id = 0; id < dim; ++ic, ++id)
    {
      forcing[ic] = parameters->gravity[id] * W[density_component];
    }

  // Nothing for mass equation


  // For energy equation, work rate = force \cdot velocity
  forcing[energy_component] = 0.0;
  for (unsigned int ic = first_velocity_component, id = 0; id < dim; ++ic, ++id)
    {
      forcing[energy_component] += forcing[ic] * W[ic];
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
template <int dim>
template <typename DataVector>
void
EulerEquations<dim>::compute_Wminus(const Boundary::Type &boundary_kind,
                                    const Tensor<1, dim> &normal_vector,
                                    const DataVector &    Wplus,
                                    const Vector<double> &boundary_values,
                                    DataVector &          Wminus)
{
  Assert(parameters, ExcMessage("Null pointer encountered!"));
  typedef typename DataVector::value_type VType;
  std::array<VType, n_components>         requested_boundary_values;
  bool                                    need_Riemman_correction = false;

  switch (boundary_kind)
    {
      case Boundary::AllPrimitiveValues:
        {
          for (unsigned int ic = 0; ic < n_components; ++ic)
            {
              requested_boundary_values[ic] = boundary_values[ic];
            }
          need_Riemman_correction = true;
          break;
        }

      case Boundary::NonSlipWall:
        {
          for (unsigned int ic = first_velocity_component, id = 0; id < dim;
               ++ic, ++id)
            {
              // This have almost same effect as Wminus[ic] = -Wplus[ic]. But
              // residual is a little bit small. I think this may be more stable
              // because Wminus has no dependency on internal value and always
              // has the desired value.
              Wminus[ic] = 0.0;
            }
          Wminus[density_component]  = Wplus[density_component];
          Wminus[pressure_component] = Wplus[pressure_component];

          break;
        }

      case Boundary::FarField:
        {
          // Here we assume x axis points backward, y axis points upward and
          // z axis points sideward.
          int const             x = 0;
          int const             y = 1;
          int const             z = 2;
          std::array<double, 3> velocity_infty;
          velocity_infty[z] =
            parameters->Mach * std::sin(parameters->angle_of_side_slip);
          double const velocity_in_symm =
            parameters->Mach * std::cos(parameters->angle_of_side_slip);
          velocity_infty[x] =
            velocity_in_symm * std::cos(parameters->angle_of_attack);
          velocity_infty[y] =
            velocity_in_symm * std::sin(parameters->angle_of_attack);

          // Extrapolate velocity and density
          for (unsigned int id = 0, ic = first_velocity_component; id < dim;
               ++ic, ++id)
            {
              requested_boundary_values[ic] = velocity_infty[id];
            }
          requested_boundary_values[density_component]  = 1.0;
          requested_boundary_values[pressure_component] = 1.0 / gas_gamma;

          need_Riemman_correction = true;
          break;
        }

      case Boundary::PressureOutlet:
        {
          // Extrapolate velocity and density
          for (unsigned int ic = first_velocity_component;
               ic < first_velocity_component + dim;
               ++ic)
            {
              requested_boundary_values[ic] = Wplus[ic];
            }
          requested_boundary_values[density_component] =
            Wplus[density_component];

          // Enforce pressure
          requested_boundary_values[pressure_component] =
            boundary_values[pressure_component];

          need_Riemman_correction = true;
          break;
        }

      case Boundary::MomentumInlet:
        {
          // Enforce velocity and density
          for (unsigned int ic = first_velocity_component;
               ic < first_velocity_component + dim;
               ++ic)
            {
              requested_boundary_values[ic] = boundary_values[ic];
            }
          requested_boundary_values[density_component] =
            boundary_values[density_component];

          // Extrapolate pressure
          requested_boundary_values[pressure_component] =
            Wplus[pressure_component];

          need_Riemman_correction = true;
          break;
        }

      case Boundary::MMS_BC:
        {
          // MMS_BC boundary condition set values of components once for all.
          // It is similar to Riemann boundary condition, but enforce all
          // boundary values in subsonic case. The reason is the solution is
          // manufactured and the in enforced, the solution has no degree of
          // freedom. However, on supersonic out flow boundary, the solution
          // still has to be extrapolated. This is because in supersonic case
          // the solution depends only on inflow boundary. Compute sound speed
          // and normal velocity from demanded boundary values
          VType const sound_speed_incoming =
            compute_sound_speed(boundary_values);
          VType normal_velocity_incoming = 0.0;
          for (unsigned int ic = first_velocity_component;
               ic < first_velocity_component + dim;
               ++ic)
            {
              normal_velocity_incoming +=
                boundary_values[ic] * normal_vector[ic];
            }

          if (normal_velocity_incoming - sound_speed_incoming >= 0.0)
            {
              // This is a supersonic outflow boundary, extrapolate all boundary
              // values without calculating characteristics values
              for (unsigned int ic = 0; ic < n_components; ++ic)
                {
                  Wminus[ic] = Wplus[ic];
                }
            }
          else
            {
              for (unsigned int ic = 0; ic < n_components; ++ic)
                {
                  Wminus[ic] = boundary_values[ic];
                }
            }
          break;
        }

      case Boundary::Symmetry:
        {
          // Project velocity onto boundary This creates sensitivities of
          // across the velocity components.
          VType vdotn = 0;
          for (unsigned int d = 0; d < dim; ++d)
            {
              vdotn += Wplus[d] * normal_vector[d];
            }
          if (parameters->laplacian_continuation > 0.0)
            {
              for (unsigned int ic = first_velocity_component, id = 0; id < dim;
                   ++ic, ++id)
                {
                  // With laplacian_continuation, boundary flux is computed from
                  // demanded boundary condition directly rather than from
                  // numerical flux. So here we need to use tangential velocity
                  // instead of mirrored velocity.
                  requested_boundary_values[ic] =
                    Wplus[ic] - vdotn * normal_vector[id];
                }
            }
          else
            {
              for (unsigned int ic = first_velocity_component, id = 0; id < dim;
                   ++ic, ++id)
                {
                  requested_boundary_values[ic] =
                    Wplus[ic] - 2.0 * vdotn * normal_vector[id];
                }
            }

          requested_boundary_values[density_component] =
            Wplus[density_component];
          requested_boundary_values[pressure_component] =
            Wplus[pressure_component];

          for (unsigned int ic = 0; ic < n_components; ++ic)
            {
              Wminus[ic] = requested_boundary_values[ic];
            }
          break;
        }

      default:
        Assert(false, ExcNotImplemented());
        break;
    }

  if (need_Riemman_correction)
    {
      // The required boundary should be applied in characteristics form.

      // Compute sound speed and normal velocity from demanded boundary values
      VType const sound_speed_incoming =
        compute_sound_speed(requested_boundary_values);

      VType normal_velocity_incoming = 0.0;
      for (unsigned int d = first_velocity_component;
           d < first_velocity_component + dim;
           ++d)
        {
          normal_velocity_incoming +=
            requested_boundary_values[d] * normal_vector[d];
        }

      if (normal_velocity_incoming + sound_speed_incoming <= 0.0)
        {
          // This is a supersonic inflow boundary, enforce all boundary values
          // without calculating characteristics values
          for (unsigned int ic = 0; ic < n_components; ++ic)
            {
              Wminus[ic] = requested_boundary_values[ic];
            }
        }
      else if (normal_velocity_incoming - sound_speed_incoming >= 0.0)
        {
          // This is a supersonic outflow boundary, extrapolate all boundary
          // values without calculating characteristics values
          for (unsigned int ic = 0; ic < n_components; ++ic)
            {
              Wminus[ic] = Wplus[ic];
            }
        }
      else
        {
          // This is a subsonic boundary. Evaluate interior normal speed and
          // sound speed.
          VType const sound_speed_outcoming = compute_sound_speed(Wplus);

          VType normal_velocity_outcoming = 0.0;
          for (unsigned int d = first_velocity_component;
               d < first_velocity_component + dim;
               ++d)
            {
              normal_velocity_outcoming += Wplus[d] * normal_vector[d];
            }
          // v_n+c > 0, thus compute v_n + 2*c/(gas_gamma - 1) from interior
          // values.
          VType riemann_invariant_outcoming =
            normal_velocity_outcoming +
            2.0 * sound_speed_outcoming / (gas_gamma - 1.0);
          // v_n-c < 0, thus compute v_n - 2*c/(gas_gamma - 1) from demanded
          // boundary values.
          VType riemann_invariant_incoming =
            normal_velocity_incoming -
            2.0 * sound_speed_incoming / (gas_gamma - 1.0);

          VType                  entropy_boundary;
          std::array<VType, dim> tangential_velocity;
          if (normal_velocity_incoming <= 0.0)
            {
              // v_n <= 0
              // This is a subsonic inflow boundary.
              // Compute tangential velocity and entropy(not exactly) from
              // demanded boundary values.
              for (unsigned int d = first_velocity_component, i = 0; i < dim;
                   ++d, ++i)
                {
                  tangential_velocity[i] =
                    requested_boundary_values[d] -
                    normal_velocity_incoming * normal_vector[d];
                }
              entropy_boundary =
                std::pow(requested_boundary_values[density_component],
                         gas_gamma) /
                requested_boundary_values[pressure_component];
            }
          else
            {
              // v_n > 0
              // This is a subsonic outflow boundary.
              // Compute tangential velocity and entropy(not exactly) from
              // interior values.
              for (unsigned int d = first_velocity_component, i = 0; i < dim;
                   ++d, ++i)
                {
                  tangential_velocity[i] =
                    Wplus[d] - normal_velocity_outcoming * normal_vector[d];
                }
              entropy_boundary = std::pow(Wplus[density_component], gas_gamma) /
                                 Wplus[pressure_component];
            }
          // Recover primitive variables from characteristics variables.
          VType const normal_velocity_boundary =
            0.5 * (riemann_invariant_outcoming + riemann_invariant_incoming);
          VType const sound_speed_boundary =
            0.25 * (gas_gamma - 1.0) *
            (riemann_invariant_outcoming - riemann_invariant_incoming);
          Assert(sound_speed_boundary > 0.0,
                 ExcLowerRangeType<VType>(sound_speed_boundary, 0.0));

          for (unsigned int d = first_velocity_component, i = 0; i < dim;
               ++d, ++i)
            {
              Wminus[d] = tangential_velocity[i] +
                          normal_velocity_boundary * normal_vector[d];
            }
          Wminus[density_component] =
            std::pow(sound_speed_boundary * sound_speed_boundary *
                       entropy_boundary / gas_gamma,
                     1.0 / (gas_gamma - 1.0));
          Wminus[pressure_component] = Wminus[density_component] *
                                       sound_speed_boundary *
                                       sound_speed_boundary / gas_gamma;
        }
    } // End Riemann correction

  return;
}

#endif
