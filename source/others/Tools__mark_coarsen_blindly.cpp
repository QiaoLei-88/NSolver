


#include <NSolver/Tools.h>

namespace NSFEMSolver
{
  using namespace dealii;
  namespace Tools
  {
    template <int dim, typename Number>
    void
    mark_coarsen_blindly (parallel::distributed::Triangulation<dim> &tria,
                          const Vector<Number>                      &criteria,
                          const std::pair<double,double>            global_min_and_max,
                          const double                              target_fraction,
                          const std::vector<short int>              &coarsen_mask)
    {
      Assert (criteria.size() == tria.n_active_cells(),
              ExcDimensionMismatch (criteria.size(), tria.n_active_cells()));
      Assert (global_min_and_max.first >= 0,
              ExcMessage ("Negative refinement criteria is not allowed."));
      Assert (global_min_and_max.first <= global_min_and_max.second,
              ExcMessage ("Min criteria should not greater than max criteria."));
      Assert (target_fraction>=0.0,
              dealii::GridRefinement::ExcInvalidParameterValue());
      Assert (target_fraction<=1.0,
              dealii::GridRefinement::ExcInvalidParameterValue());
      Assert (coarsen_mask.size() == tria.n_active_cells(),
              ExcDimensionMismatch (coarsen_mask.size(), tria.n_active_cells()));

      typedef parallel::distributed::Triangulation<dim> TypeTria;

      //Short cuts
      // Coarsen none
      if (target_fraction <= 0.0)
        {
          return;
        }
      // Coarsen all
      if (target_fraction >= 1.0)
        {
          dealii::GridRefinement::coarsen (tria, criteria, global_min_and_max.second + 1.0);
          return;
        }
      // All same criteria
      if (global_min_and_max.first == global_min_and_max.second)
        {
          // Mark cell in a greedy manner
          dealii::GridRefinement::coarsen (tria, criteria, global_min_and_max.first);
          return;
        }

      const MPI_Comm mpi_communicator = tria.get_communicator();
      const unsigned int master_mpi_rank = 0;

      const unsigned int n_target_cells =
        target_fraction * tria.n_global_active_cells();

      // The mark-all and mark-none cases are handle above, the result threshold
      // can never lay on the extreme value of indicator. so it is safe to use
      // the exact indicator range as initial bisection search range.
      double interesting_range[2] = { global_min_and_max.first,
                                      global_min_and_max.second
                                    };
      // Bisection search
      unsigned max_n_loop = 3;
      for (unsigned int error_range = tria.n_global_active_cells();
           error_range != 0;
           error_range /= 2)
        {
          ++max_n_loop;
        }
      double test_threshold = 0.5 * (interesting_range[0] + interesting_range[1]);
      for (unsigned int n = 0; n<max_n_loop; ++n)
        {
          test_threshold = 0.5 * (interesting_range[0] + interesting_range[1]);
          unsigned int my_count = 0;
          for (typename TypeTria::active_cell_iterator cell = tria.begin_active();
               cell != tria.end();
               ++cell)
            if (cell->is_locally_owned())
              {
                bool will_be_coarsened = false;
                if (coarsen_mask[cell->active_cell_index()] == 0)
                  {
                    will_be_coarsened =
                      (criteria[cell->active_cell_index()] < test_threshold);
                  }
                else
                  {
                    will_be_coarsened =
                      (coarsen_mask[cell->active_cell_index()] > 0);
                  }
                // bool to int conversion is standard conformant
                my_count += will_be_coarsened;
              }
          unsigned int total_count;
          MPI_Reduce (&my_count, &total_count, 1, MPI_UNSIGNED,
                      MPI_SUM, master_mpi_rank, mpi_communicator);
          if (total_count >= n_target_cells)
            {
              interesting_range[1] = test_threshold;
            }
          if (total_count <= n_target_cells)
            {
              interesting_range[0] = test_threshold;
            }
          MPI_Bcast (&interesting_range[0], 2, MPI_DOUBLE,
                     master_mpi_rank, mpi_communicator);
          if (interesting_range[0] == interesting_range[1])
            {
              break;
            }
        }

      dealii::GridRefinement::coarsen (tria, criteria, test_threshold);
      return;
    }

    // Instantiation
    template
    void
    mark_coarsen_blindly<2, float> (parallel::distributed::Triangulation<2> &,
                                    const Vector<float> &,
                                    const std::pair<double,double>,
                                    const double,
                                    const std::vector<short int> &);
    template
    void
    mark_coarsen_blindly<2, double> (parallel::distributed::Triangulation<2> &,
                                     const Vector<double> &,
                                     const std::pair<double,double>,
                                     const double,
                                     const std::vector<short int> &);
    template
    void
    mark_coarsen_blindly<3, float> (parallel::distributed::Triangulation<3> &,
                                    const Vector<float> &,
                                    const std::pair<double,double>,
                                    const double,
                                    const std::vector<short int> &);
    template
    void
    mark_coarsen_blindly<3, double> (parallel::distributed::Triangulation<3> &,
                                     const Vector<double> &,
                                     const std::pair<double,double>,
                                     const double,
                                     const std::vector<short int> &);
  }
}
