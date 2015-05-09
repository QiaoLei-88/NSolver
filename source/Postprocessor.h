//
//  Created by 乔磊 on 15/4/24.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//

#ifndef __NSolver__Postprocessor__
#define __NSolver__Postprocessor__


#include <deal.II/numerics/data_postprocessor.h>
#include <deal.II/numerics/data_component_interpretation.h>
#include "NSEquation.h"
#include "AllParameters.h"
#include "MMS.h"

namespace NSolver
{
  using namespace dealii;

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
  template <int dim>
  class Postprocessor : public DataPostprocessor<dim>
  {
  public:
    Postprocessor (Parameters::AllParameters<dim> const *const para_ptr_in,
                   MMS const *const mms_ptr_in);

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
    Parameters::AllParameters<dim>const *const parameters;
    MMS const *const mms_x;
    bool const do_schlieren_plot;
    bool const output_mms;
  };
}
#endif
