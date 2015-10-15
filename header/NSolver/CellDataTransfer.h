//
//  CellDataTransfer.h
//  NSolver
//
//  Created by 乔磊 on 15/10/13.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//

#ifndef NSolver_CellDataTransfer_h
#define NSolver_CellDataTransfer_h

#include <deal.II/grid/tria.h>
#include <deal.II/base/std_cxx11/bind.h>

namespace NSFEMSolver
{
  using namespace dealii;

  template<int dim, typename Number = double>
  class CellDataTransfer
  {
  public:
    typedef unsigned int size_type;

    /**
     * The constructor first set all user indices on active cells if @p tria_in
     * to @p active_cell_index, then connect proper member function to
     * corresponding signals of @p tria_in. So that data attached to active cells
     * could be transfered during cell refinement and coarsening.
     */
    CellDataTransfer (Triangulation<dim> &tria_in);

    /**
     * Destructor, clean up all internal data caches.
     */
    ~CellDataTransfer();

    /**
     * Receive a vector via starting address @p data_src and length @size
     * so that its contents could be transfer the new mesh during mesh adaptation.
     * The vector handed in will be assigned an index automatically witch is the
     * return value of this function.
     * The index starts from 0 and increases 1 after another so assign multiple
     * vectors is allowed and value types of the vectors don't have to be same.
     * You need to provide correct index to get_transfered_data() to retrieve
     * the transfered data.
     *
     * @note Value's in the provided vector is assume in the order that same to
     * active cells.
     *
     * @note Value of @p size should be no less than @p tria.n_active_cell().
     * @p tria is the triangulation used to initialize the current object. All
     * extra data in @p data_src will be ignored. Size check only performed in
     * DEBUG mode.
     *
     * @note It is safe to destroy vector @p data_src after this function call.
     * All its data in range [0, tria.n_active_cell()) will be cached internally.
     *
     * @note Instantiated ValueType types are <tt>double</tt> and <tt>float</tt>.
     * All the data will be implicitly converted to and store internally in
     * type <tt>Number</tt>.
     */
    template <typename ValueType>
    size_type push_back (ValueType *const data_src, const size_type size);

    /**
     * Retrieve transfered data if index @index to vector with starting address @p data_src.
     *
     * @note Please make the receiving vector has enough space. No check is provided
     * in this function.
     *
     * @note You can receive one vector multiple times. This is not a pop operation.
     *
     * @note Variable type of the receiving vector does not have to be same to
     * the type of Number or corresponding provided vector. Implicit type conversion
     * applies.
     */
    template <typename ValueType>
    void get_transfered_data (const unsigned int index, ValueType *data_dest) const;

    /**
     * Free all internally allocated memory. You can call this function
     * immediately after data transfer finished.
     */
    void clear();

  private:

    /**
     * Set data on all children cells according to data on parent cell @p cell.
     * This function is connected to Triangulation::signals.pre_coarsening_on_cell.
     */
    virtual void parent_to_children (const typename Triangulation<dim>::cell_iterator &cell);

    /**
     * Set data on parent cell @p cell according to data on children cells.
     * This function is connected to Triangulation::signals.post_refinement_on_cell.
     */
    virtual void children_to_parent (const typename Triangulation<dim>::cell_iterator &cell);

    /**
     * Cell could be created, deleted, activated or deactivated during adaptation.
     * New memory will be allocated for data corresponds to newly created or
     * activated cells, but data corresponds to deleted or deactivated cells will
     * not be destroyed immediately.
     * This counter is used to keep track on the total number of active data.
     * This is not necessary for data transferring, only used for safety checks.
     */
    size_type active_data_size;

    /**
     * Internal cache for all handed-in data and transfered data.
     */
    std::vector<std::vector<Number> > vector_data_ptr;

    /**
     * A constant reference to the related triangulation that used to loop through
     * all cells.
     */
    const Triangulation<dim> &tria;

    /**
     * Signal connections between the related triangulation. We need to store them
     * in order to disconnect before destruction.
     */
    boost::signals2::connection refine_listener;
    boost::signals2::connection coarsen_listener;
  };
}
#endif
