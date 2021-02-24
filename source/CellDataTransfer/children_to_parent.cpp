//
//  CellDataTransfer::children_to_parent.cpp
//  NSolver
//
//  Created by 乔磊 on 15/10/13.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//

#include <NSolver/CellDataTransfer.h>

namespace NSFEMSolver
{
  template <int dim, typename InternalValueType>
  void
  CellDataTransfer<dim, InternalValueType>::children_to_parent(
    const typename Triangulation<dim>::cell_iterator &cell)
  {
    // Collect children values to parent cell
    const size_type n_vector = vector_data_ptr.size();
    if (n_vector == 0)
      {
        // Nothing to do when no vector attached.
        return;
      }

    std::vector<InternalValueType> children_sum(n_vector, 0.0);
    for (unsigned int i_child = 0; i_child < cell->n_children(); ++i_child)
      {
        Assert(cell->child(i_child)->is_active(),
               ExcMessage(
                 "A cell is going to coarsen should have active children."));
        const size_type child_index = cell->child(i_child)->user_index();
        for (size_type i = 0; i < n_vector; ++i)
          {
            children_sum[i] += vector_data_ptr[i][child_index];
          }
        --active_data_size;
      }
    const size_type next_position = vector_data_ptr[0].size();
    // Store values of parent cell
    for (size_type i = 0; i < n_vector; ++i)
      {
        vector_data_ptr[i].push_back(children_sum[i]);
      }
    // Tell the cell where is its data
    cell->set_user_index(next_position);
    ++active_data_size;

    return;
  }

#include "CellDataTransfer.inst"
} // namespace NSFEMSolver
