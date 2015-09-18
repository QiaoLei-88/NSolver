


#include <NSolver/Tools.h>

namespace NSFEMSolver
{
  using namespace dealii;
  namespace Tools
  {
    template <int dim, typename Number>
    void
    mark_coarsen_guaranteed (parallel::distributed::Triangulation<dim> &tria,
                             const Vector<Number>                      &criteria,
                             const std::pair<double,double>            global_min_and_max,
                             const unsigned int                        target_n_cell_drop,
                             const std::vector<short int>              &coarsen_mask)
    {
      Assert (criteria.size() == tria.n_active_cells(),
              ExcDimensionMismatch (criteria.size(), tria.n_active_cells()));
      Assert (global_min_and_max.first >= 0,
              ExcMessage ("Negative refinement criteria is not allowed."));
      Assert (global_min_and_max.first <= global_min_and_max.second,
              ExcMessage ("Min criteria should not greater than max criteria."));
      Assert (coarsen_mask.size() == tria.n_active_cells(),
              ExcDimensionMismatch (coarsen_mask.size(), tria.n_active_cells()));

      typedef parallel::distributed::Triangulation<dim> TypeTria;
      //Short cuts
      // Coarsen none
      if (target_n_cell_drop == 0)
        {
          typename TypeTria::active_cell_iterator
          cell = tria.begin_active();
          const typename TypeTria::active_cell_iterator
          endc = tria.end();
          for (; cell != endc; ++cell)
            if (cell->is_locally_owned())
              {
                cell->clear_coarsen_flag();
              }
          return;
        }
      // Coarsen all
      if (target_n_cell_drop >=
          ((tria.n_global_active_cells() * (GeometryInfo<dim>::max_children_per_cell - 1))
           / GeometryInfo<dim>::max_children_per_cell))
        {
          typename TypeTria::active_cell_iterator
          cell = tria.begin_active();
          const typename TypeTria::active_cell_iterator
          endc = tria.end();
          for (; cell != endc; ++cell)
            if (cell->is_locally_owned())
              {
                cell->clear_refine_flag();
                cell->set_coarsen_flag();
              }
          return;
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

      // save status of refine_flags
      std::vector<bool> saved_refine_flags (tria.n_active_cells());
      {
        typename TypeTria::active_cell_iterator
        cell = tria.begin_active();
        const typename TypeTria::active_cell_iterator
        endc = tria.end();
        for (; cell != endc; ++cell)
          if (cell->is_locally_owned())
            {
              saved_refine_flags[cell->active_cell_index()] =
                cell->refine_flag_set();
            }
      }

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
                  cell->clear_coarsen_flag();
                  if (saved_refine_flags[cell->active_cell_index()])
                    {
                      cell->set_refine_flag();
                    }
                }
          }
          // try refine and coarsen flags
          {
            typename TypeTria::active_cell_iterator
            cell = tria.begin_active();
            const typename TypeTria::active_cell_iterator
            endc = tria.end();
            for (; cell != endc; ++cell)
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
                  if (will_be_coarsened)
                    {
                      cell->clear_refine_flag();
                      cell->set_coarsen_flag();
                    }
                }
          }
          // Do smoothing
          tria.prepare_coarsening_and_refinement();
          // count cell number drop
          unsigned int n_cell_marked = 0;
          {
            typename TypeTria::active_cell_iterator
            cell = tria.begin_active();
            const typename TypeTria::active_cell_iterator
            endc = tria.end();
            for (; cell != endc; ++cell)
              if (cell->is_locally_owned())
                {
                  if (cell->coarsen_flag_set())
                    {
                      ++n_cell_marked;
                    }
                  if (saved_refine_flags[cell->active_cell_index()]
                      ^
                      static_cast<bool> (cell->refine_flag_set()))
                    {
                      n_cell_marked += GeometryInfo<dim>::max_children_per_cell;
                    }
                }
          }
          // Because counting of n_cell_marked is after smoothing, there should
          // be not truncation in n_cell_drop_local.
          unsigned int n_cell_drop_local =
            (n_cell_marked * (GeometryInfo<dim>::max_children_per_cell - 1))
            / GeometryInfo<dim>::max_children_per_cell;

          unsigned int total_n_cell_drop;
          MPI_Reduce (&n_cell_drop_local, &total_n_cell_drop, 1, MPI_UNSIGNED,
                      MPI_SUM, master_mpi_rank, mpi_communicator);
          if (total_n_cell_drop <= target_n_cell_drop)
            {
              interesting_range[0] = test_threshold;
            }
          if (total_n_cell_drop >= target_n_cell_drop)
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
      return;
    }

    // Instantiation
    template
    void
    mark_coarsen_guaranteed<2, float> (parallel::distributed::Triangulation<2> &,
                                       const Vector<float> &,
                                       const std::pair<double,double>,
                                       const unsigned int,
                                       const std::vector<short int> &);
    template
    void
    mark_coarsen_guaranteed<2, double> (parallel::distributed::Triangulation<2> &,
                                        const Vector<double> &,
                                        const std::pair<double,double>,
                                        const unsigned int,
                                        const std::vector<short int> &);
    template
    void
    mark_coarsen_guaranteed<3, float> (parallel::distributed::Triangulation<3> &,
                                       const Vector<float> &,
                                       const std::pair<double,double>,
                                       const unsigned int,
                                       const std::vector<short int> &);
    template
    void
    mark_coarsen_guaranteed<3, double> (parallel::distributed::Triangulation<3> &,
                                        const Vector<double> &,
                                        const std::pair<double,double>,
                                        const unsigned int,
                                        const std::vector<short int> &);
  } // NAMESPACE Tools
} // NAMESPACE NSFEMSolver
