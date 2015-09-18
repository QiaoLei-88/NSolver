


#include <NSolver/Tools.h>

namespace NSFEMSolver
{
  using namespace dealii;
  namespace Tools
  {
    template <int dim, typename Number>
    unsigned int
    mark_refine_guaranteed (parallel::distributed::Triangulation<dim> &tria,
                            const Vector<Number>                      &criteria,
                            const std::pair<double,double>            global_min_and_max,
                            const double                              target_fraction,
                            const std::vector<short int>              &refine_mask)
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
      Assert (refine_mask.size() == tria.n_active_cells(),
              ExcDimensionMismatch (refine_mask.size(), tria.n_active_cells()));

      typedef parallel::distributed::Triangulation<dim> TypeTria;
      const unsigned int n_target_cells =
        static_cast<unsigned int> (target_fraction * tria.n_global_active_cells() + 0.5);

      //Short cuts
      // Mark none
      if (n_target_cells == 0)
        {
          typename TypeTria::active_cell_iterator
          cell = tria.begin_active();
          const typename TypeTria::active_cell_iterator
          endc = tria.end();
          for (; cell != endc; ++cell)
            if (cell->is_locally_owned())
              {
                cell->clear_refine_flag();
              }

          return (n_target_cells);
        }
      // Mark all
      if (n_target_cells >= tria.n_global_active_cells())
        {
          typename TypeTria::active_cell_iterator
          cell = tria.begin_active();
          const typename TypeTria::active_cell_iterator
          endc = tria.end();
          for (; cell != endc; ++cell)
            if (cell->is_locally_owned())
              {
                cell->set_refine_flag();
                cell->clear_coarsen_flag();
              }

          return (n_target_cells);
        }

      // Here we go
      const MPI_Comm mpi_communicator = tria.get_communicator();
      const unsigned int master_mpi_rank = 0;

      // The mark-all and mark-none cases are handle above, the result threshold
      // can never lay on the extreme value of indicator. so it is safe to use
      // the exact indicator range as initial bisection search range.
      double interesting_range[2] = { global_min_and_max.first,
                                      global_min_and_max.second
                                    };

      unsigned int total_count = 0;

      // Bisection search
      unsigned max_n_loop = 3;
      for (unsigned int error_range = tria.n_global_active_cells();
           error_range != 0;
           error_range /= 2)
        {
          ++max_n_loop;
        }
      for (unsigned int n = 0; n<max_n_loop; ++n)
        {
          const double test_threshold
            = 0.5 * (interesting_range[0] + interesting_range[1]);

          // reset refine flags
          {
            typename TypeTria::active_cell_iterator
            cell = tria.begin_active();
            const typename TypeTria::active_cell_iterator
            endc = tria.end();
            for (; cell != endc; ++cell)
              if (cell->is_locally_owned())
                {
                  cell->clear_refine_flag();
                }
          }
          // trial marking for refinement
          {
            typename TypeTria::active_cell_iterator
            cell = tria.begin_active();
            const typename TypeTria::active_cell_iterator
            endc = tria.end();
            for (; cell != endc; ++cell)
              if (cell->is_locally_owned())
                {
                  bool will_refine = false;
                  if (refine_mask[cell->active_cell_index()] == 0)
                    {
                      will_refine =
                        (criteria[cell->active_cell_index()] > test_threshold);
                    }
                  else
                    {
                      will_refine =
                        (refine_mask[cell->active_cell_index()] > 0);
                    }
                  if (will_refine)
                    {
                      cell->set_refine_flag();
                    }
                }
          }
          // Do smoothing
          tria.prepare_coarsening_and_refinement();

          // count how many of our own elements would be refined
          unsigned int my_count = 0;
          {
            typename TypeTria::active_cell_iterator
            cell = tria.begin_active();
            const typename TypeTria::active_cell_iterator
            endc = tria.end();
            for (; cell != endc; ++cell)
              if (cell->is_locally_owned())
                {
                  if (cell->refine_flag_set())
                    {
                      ++my_count;
                    }
                }
          }

          MPI_Reduce (&my_count, &total_count, 1, MPI_UNSIGNED,
                      MPI_SUM, master_mpi_rank, mpi_communicator);
          if (total_count >= n_target_cells)
            {
              interesting_range[0] = test_threshold;
            }
          if (total_count <= n_target_cells)
            {
              interesting_range[1] = test_threshold;
            }
          MPI_Bcast (&interesting_range[0], 2, MPI_DOUBLE,
                     master_mpi_rank, mpi_communicator);
          if (interesting_range[1] == interesting_range[0])
            {
              break;
            }
        } // bisection iteration
      MPI_Bcast (&total_count, 1, MPI_UNSIGNED, master_mpi_rank, mpi_communicator);
      return (total_count);
    }

    // Instantiation
    template
    unsigned int
    mark_refine_guaranteed<2, float> (parallel::distributed::Triangulation<2> &,
                                      const Vector<float> &,
                                      const std::pair<double,double>,
                                      const double,
                                      const std::vector<short int> &);
    template
    unsigned int
    mark_refine_guaranteed<2, double> (parallel::distributed::Triangulation<2> &,
                                       const Vector<double> &,
                                       const std::pair<double,double>,
                                       const double,
                                       const std::vector<short int> &);
    template
    unsigned int
    mark_refine_guaranteed<3, float> (parallel::distributed::Triangulation<3> &,
                                      const Vector<float> &,
                                      const std::pair<double,double>,
                                      const double,
                                      const std::vector<short int> &);
    template
    unsigned int
    mark_refine_guaranteed<3, double> (parallel::distributed::Triangulation<3> &,
                                       const Vector<double> &,
                                       const std::pair<double,double>,
                                       const double,
                                       const std::vector<short int> &);

  } // NAMESPACE Tools
} // NAMESPACE NSFEMSolver
