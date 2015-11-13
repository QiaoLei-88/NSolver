


#include <NSolver/Tools.h>

namespace NSFEMSolver
{
  using namespace dealii;
  namespace Tools
  {
    template <int dim, class Vector>
    void
    refine_and_coarsen_fixed_number (
      parallel::distributed::Triangulation<dim>           &tria,
      const Vector                                        &criteria,
      const SmartPointer<Parameters::AllParameters<dim> const> &parameters)
    {
      typedef typename parallel::distributed::Triangulation<dim> TypeTria;
      typedef typename Vector::value_type VType;
      const double top_fraction_of_cells = parameters->refine_fraction;
      const double bottom_fraction_of_cells = parameters->coarsen_fraction;

      Assert (criteria.size() == tria.n_active_cells(),
              ExcDimensionMismatch (criteria.size(), tria.n_active_cells()));
      Assert (top_fraction_of_cells>=0.0,
              dealii::GridRefinement::ExcInvalidParameterValue());
      Assert (top_fraction_of_cells<=1.0,
              dealii::GridRefinement::ExcInvalidParameterValue());
      Assert (bottom_fraction_of_cells>=0.0,
              dealii::GridRefinement::ExcInvalidParameterValue());
      Assert (bottom_fraction_of_cells<=1.0,
              dealii::GridRefinement::ExcInvalidParameterValue());
      Assert (top_fraction_of_cells+bottom_fraction_of_cells <= 1.0,
              dealii::GridRefinement::ExcInvalidParameterValue());
      Assert (criteria.is_non_negative(),
              dealii::GridRefinement::ExcNegativeCriteria());

      const MPI_Comm mpi_communicator = tria.get_communicator();

      // figure out the global max and min of the indicators.
      std::pair<double,double> global_min_and_max;
      {
        typename TypeTria::active_cell_iterator
        cell = tria.begin_active();
        const typename TypeTria::active_cell_iterator
        endc = tria.end();
        VType local_min = std::numeric_limits<VType>::max();
        VType local_max = std::numeric_limits<VType>::min();
        for (; cell!=endc; ++cell)
          if (cell->is_locally_owned())
            {
              local_min = std::min (local_min,
                                    criteria[cell->active_cell_index()]);
              local_max = std::max (local_max,
                                    criteria[cell->active_cell_index()]);
            }
        global_min_and_max.first  =
          static_cast<double> (Utilities::MPI::min (local_min, mpi_communicator));
        global_min_and_max.second =
          static_cast<double> (Utilities::MPI::max (local_max, mpi_communicator));
      }

      std::vector<short int> refine_mask (tria.n_active_cells(), 0);
      set_refine_mask (tria, parameters, refine_mask);

      std::vector<short int> coarsen_mask (tria.n_active_cells(), 0);
      set_coarsen_mask (tria, parameters, coarsen_mask);

      const unsigned int n_refined =
        mark_refine_guaranteed<dim, VType> (tria,
                                            criteria,
                                            global_min_and_max,
                                            top_fraction_of_cells,
                                            refine_mask);

      const unsigned int cell_increase_on_refine =
        GeometryInfo<dim>::max_children_per_cell - 1;
      const unsigned int n_cell_after_refine =
        tria.n_global_active_cells() + cell_increase_on_refine * n_refined;

      if (n_cell_after_refine > parameters->max_cells)
        {
          // If the planed refinement is going to break the max cell number
          // limit, we need to make sure the effectiveness of coarsening marks
          // is guaranteed.
          const unsigned int n_cell_exceeded =
            n_cell_after_refine - parameters->max_cells;
          mark_coarsen_guaranteed<dim, VType> (tria,
                                               criteria,
                                               global_min_and_max,
                                               n_cell_exceeded,
                                               coarsen_mask);
        }
      else
        {
          // Otherwise, if the planed refinement is not going to break the max cell
          // number limit, we just need to mark cell to coarsen blindly, no matter
          // the cell will really coarsen or not.
          mark_coarsen_blindly<dim, VType> (tria,
                                            criteria,
                                            global_min_and_max,
                                            bottom_fraction_of_cells,
                                            coarsen_mask);
        }
      return;
    }

    // Instantiation
    template
    void
    refine_and_coarsen_fixed_number<2, Vector<float> > (
      parallel::distributed::Triangulation<2> &,
      const Vector<float> &,
      const SmartPointer<Parameters::AllParameters<2> const> &);

    template
    void
    refine_and_coarsen_fixed_number<2, Vector<double> > (
      parallel::distributed::Triangulation<2> &,
      const Vector<double> &,
      const SmartPointer<Parameters::AllParameters<2> const> &);

    template
    void
    refine_and_coarsen_fixed_number<3, Vector<float> > (
      parallel::distributed::Triangulation<3> &,
      const Vector<float> &,
      const SmartPointer<Parameters::AllParameters<3> const> &);

    template
    void
    refine_and_coarsen_fixed_number<3, Vector<double> > (
      parallel::distributed::Triangulation<3> &,
      const Vector<double> &,
      const SmartPointer<Parameters::AllParameters<3> const> &);

  } // NAMESPACE Tools
} // NAMESPACE NSFEMSolver
