
#ifndef __FEMNSolver__Tools__
#define __FEMNSolver__Tools__

#include <deal.II/base/smartpointer.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_refinement.h>

#include <deal.II/lac/vector.h>

#include <NSolver/Parameters/AllParameters.h>
#include <NSolver/types.h>

namespace NSFEMSolver
{
  using namespace dealii;
  namespace Tools
  {
    template <typename Matrix>
    void
    write_matrix_MTX(std::ostream &out, const Matrix &matrix);

    /**
     * Set up refine mask vector for triangulation @p tria according to runtime
     * parameter @parameters.
     *
     * The meaning of values in @p refine_mask are defined as following.
     * @p refine_mask[i] > 0 means cell @p i is forced to refine,
     * @p refine_mask[i] < 0 means cell @p i is forced to not refine,
     * @p refine_mask[i] == 0 means refinement cell @p i is determined by
     * its value in @p criteria.
     *
     * Values in @p refine_mask will be stored in active_cell_index order.
     *
     * @note @p refine_mask is expected has already been sized to tria.n_active_cells().
     */
    template <int dim>
    void
    set_refine_mask(
      const parallel::distributed::Triangulation<dim> &         tria,
      const SmartPointer<Parameters::AllParameters<dim> const> &parameters,
      std::vector<short int> &                                  refine_mask);

    /**
     * Mark @p target_fraction fraction number of cells to refine.
     * The result is generated with consideration of deal.II mesh smoothing and
     * @p refine_mask.
     *
     * The return value is the cell number that are finally marked to refine.
     *
     * Because deal.II mesh smoothing is considered, marked cells are guaranteed
     * to refine. However, because p4est mesh smoothing is not considered, and
     * p4est has known issue about extra cell refinements, the actual cell
     * refine number may be more than the return value.
     *
     * @note If @p criteria has a fairly flat distribution, i.e., most of cells
     * have the same @p criteria value, this function will not perform well.
     */
    template <int dim, typename Number>
    unsigned int
    mark_refine_guaranteed(parallel::distributed::Triangulation<dim> &tria,
                           const Vector<Number> &                     criteria,
                           const std::pair<double, double> global_min_and_max,
                           const double                    target_fraction,
                           const std::vector<short int> &  refine_mask);

    /**
     * Set up coarsen mask vector for triangulation @p tria according to runtime
     * parameter @parameters.
     *
     * The meaning of values in @p coarsen_mask are defined as following.
     * @p coarsen_mask[i] > 0 means cell @p i is forced to coarsen,
     * @p coarsen_mask[i] < 0 means cell @p i is forced to not coarsen,
     * @p coarsen_mask [i] == 0 means refinement cell @p i is determined by
     * its value in @p criteria.
     *
     * Values in @p refine_mask will be stored in active_cell_index order.
     *
     * @note @p refine_mask is expected has already been sized to tria.n_active_cells().
     */
    template <int dim>
    void
    set_coarsen_mask(
      const parallel::distributed::Triangulation<dim> &         tria,
      const SmartPointer<Parameters::AllParameters<dim> const> &parameters,
      std::vector<short int> &                                  coarsen_mask);

    /**
     * Mark enough number of cell to coarsen to achieve a total cell number
     * decrease of @p target_n_cell_drop.
     * The result is generated with consideration of deal.II mesh smoothing and
     * @p coarsen_mask.
     *
     * Because deal.II mesh smoothing is considered, marked cells are guaranteed
     * to coarsen.
     *
     * @note Effect of canceling cell refinement is counted into cell number
     * drop. So this function should be called after mark_refine_guaranteed().
     *
     * @note If @p criteria has a fairly flat distribution, i.e., most of cells
     * have the same @p criteria value, this function will not perform well.
     */
    template <int dim, typename Number>
    void
    mark_coarsen_guaranteed(parallel::distributed::Triangulation<dim> &tria,
                            const Vector<Number> &                     criteria,
                            const std::pair<double, double> global_min_and_max,
                            const unsigned int              target_n_cell_drop,
                            const std::vector<short int> &  coarsen_mask);

    /**
     * Mark cell to coarsen with consideration of number fraction @p target_fraction
     * and @p coarsen_mask only. No mesh smoothing effect is considered. Because
     * in several cases cell coarsening will be canceled by deal.II mesh
     * smoothing, the cells marked by this function are not guaranteed to
     * coarsen.
     * Values in @p coarsen_mask are assumed to be stored in
     * active_cell_index order.  @p coarsen_mask[i] > 0 means cell @p i is forced
     * to coarsen, @p coarsen_mask[i] < 0 means cell @p i is forced to not coarsen,
     * @p coarsen_mask [i] == 0 means refinement cell @p i is determined by
     * its value in @p criteria.
     *
     * Because deal.II mesh smoothing is not considered, this function is a
     * little bit faster than mark_coarsen_guaranteed().
     */
    template <int dim, typename Number>
    void
    mark_coarsen_blindly(parallel::distributed::Triangulation<dim> &tria,
                         const Vector<Number> &                     criteria,
                         const std::pair<double, double> global_min_and_max,
                         const double                    target_fraction,
                         const std::vector<short int> &  coarsen_mask);

    /**
     * Mark fraction that specified in @p parameters of cells to refine and
     * coarsen according to error indicator @p criteria,
     * and max cell number, max and min cell size, max refine level limits
     * specified in @p parameters.
     *
     * First, the requested refine fraction is marked with function
     * mark_refine_guaranteed. After this, if cell number is going to exceed the
     * max cell number limit, then proper fraction of cell will be marked to
     * coarsen by function mark_coarsen_guaranteed to keep total cell number
     * under the limit. Otherwise, coarsening marking will be done with function
     * mark_coarsen_blindly.
     *
     * @note max cell number limit is not guaranteed. Although deal.II mesh smoothing
     * is taken in to consider, however the p4est mesh smoothing is not under
     * control. Beside this, usually it is impossible to find refine and coarsen
     * thresholds that result a cell number exactly equal to the desired value.
     */
    template <int dim, class Vector>
    void
    refine_and_coarsen_fixed_number(
      parallel::distributed::Triangulation<dim> &               tria,
      const Vector &                                            criteria,
      const SmartPointer<Parameters::AllParameters<dim> const> &parameters);

    /**
     * Mark cell with criteria equal or greater than threshold to refine,
     * then coarsen appropriate number of cells to maintain the max cell
     * number limit.
     *
     * @note refine and coarsening limits specified by @p parameters are considered,
     * such as max and min cells size, max refine level.
     */
    template <int dim, class Vector>
    void
    refine_on_threshold(
      parallel::distributed::Triangulation<dim> &               tria,
      const Vector &                                            criteria,
      const SmartPointer<Parameters::AllParameters<dim> const> &parameters,
      const typename Vector::value_type                         threshold);
  } // namespace Tools
} // namespace NSFEMSolver
#endif
