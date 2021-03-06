//
//  CellDataTransfer::parent_to_children.cpp
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
  CellDataTransfer<dim, InternalValueType>::parent_to_children(
    const typename Triangulation<dim>::cell_iterator &cell)
  {
    // Distribute parent values to children
    const size_type n_vector = vector_data_ptr.size();
    if (n_vector == 0)
      {
        // Nothing to do when no vector attached.
        return;
      }

    const size_type                parent_index   = cell->user_index();
    const InternalValueType        parent_measure = cell->measure();
    std::vector<InternalValueType> parent_mean(n_vector);
    for (size_type i = 0; i < n_vector; ++i)
      {
        parent_mean[i] = vector_data_ptr[i][parent_index] / parent_measure;
      }
    for (unsigned int i_child = 0; i_child < cell->n_children(); ++i_child)
      {
        Assert(cell->child(i_child)->is_active(),
               ExcMessage("A just refined cell should have active children."));
        const size_type         next_position = vector_data_ptr[0].size();
        const InternalValueType child_measure = cell->child(i_child)->measure();
        // Store values of child cells
        for (size_type i = 0; i < n_vector; ++i)
          {
            vector_data_ptr[i].push_back(child_measure * parent_mean[i]);
          }
        // Tell the cell where is its data
        cell->child(i_child)->set_user_index(next_position);
        ++active_data_size;
      }
    --active_data_size;

    return;
  }

#include "CellDataTransfer.inst"
} // namespace NSFEMSolver
