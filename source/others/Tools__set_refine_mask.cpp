



#include <NSolver/Tools.h>

namespace NSFEMSolver
{
  using namespace dealii;
  namespace Tools
  {
    template <int dim>
    void
    set_refine_mask(
      const parallel::distributed::Triangulation<dim> &         tria,
      const SmartPointer<Parameters::AllParameters<dim> const> &parameters,
      std::vector<short int> &                                  refine_mask)
    {
      typedef typename parallel::distributed::Triangulation<dim> TypeTria;

      Assert(refine_mask.size() == tria.n_active_cells(),
             ExcMessage("unexpected refine_mask size."));

      typename TypeTria::active_cell_iterator       cell = tria.begin_active();
      const typename TypeTria::active_cell_iterator endc = tria.end();
      for (; cell != endc; ++cell)
        if (cell->is_locally_owned())
          {
            if (cell->minimum_vertex_distance() > parameters->max_cell_size)
              {
                // must refine
                refine_mask[cell->active_cell_index()] = 1;
                continue;
              }
            if (cell->minimum_vertex_distance() <
                parameters->min_cell_size * 2.0)
              {
                // must not refine
                refine_mask[cell->active_cell_index()] = -1;
                continue;
              }
            if (cell->level() >= parameters->max_refine_level)
              {
                // must not refine
                refine_mask[cell->active_cell_index()] = -1;
                continue;
              }
          }
      return;
    }

    // Instantiation
    template void
    set_refine_mask(
      const parallel::distributed::Triangulation<2> &         tria,
      const SmartPointer<Parameters::AllParameters<2> const> &parameters,
      std::vector<short int> &                                refine_mask);
    template void
    set_refine_mask(
      const parallel::distributed::Triangulation<3> &         tria,
      const SmartPointer<Parameters::AllParameters<3> const> &parameters,
      std::vector<short int> &                                refine_mask);

  } // NAMESPACE Tools
} // NAMESPACE NSFEMSolver
