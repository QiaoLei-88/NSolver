//
//  CellDataTransfer::get_transfered_data.cpp
//  NSolver
//
//  Created by 乔磊 on 15/10/13.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//

#include <NSolver/CellDataTransfer.h>

namespace NSFEMSolver
{

  template<int dim, typename InternalValueType>
  template <typename UserValueType>
  void CellDataTransfer<dim, InternalValueType>::get_transfered_data (
    const unsigned int index,
    UserValueType     *data_dest) const
  {
    Assert (index < vector_data_ptr.size(),
            ExcMessage ("You are asking for data that never turned in"));
    Assert (! (active_data_size < tria.n_active_cells()),
            ExcMessage ("Tracked active cell number is less than actual value"));
    Assert (! (active_data_size > tria.n_active_cells()),
            ExcMessage ("Tracked active cell number is greater than actual value"));

    const typename Triangulation<dim>::active_cell_iterator
    endc = tria.end();
    typename Triangulation<dim>::active_cell_iterator
    cell = tria.begin_active();
    for (; cell != endc; ++cell, ++data_dest)
      {
        *data_dest = vector_data_ptr[index][cell->user_index()];
      }

    return;
  }

// Explicit instantiation
  template
  void CellDataTransfer<2, double>::
  get_transfered_data<double> (const unsigned int index,
                               double            *data_dest) const;
  template
  void CellDataTransfer<2, double>::
  get_transfered_data<float> (const unsigned int index,
                              float             *data_dest) const;
  template
  void CellDataTransfer<3, double>::
  get_transfered_data<double> (const unsigned int index,
                               double            *data_dest) const;
  template
  void CellDataTransfer<3, double>::
  get_transfered_data<float> (const unsigned int index,
                              float             *data_dest) const;

#include "CellDataTransfer.inst"
}
