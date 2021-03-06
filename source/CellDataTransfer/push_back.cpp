//
//  CellDataTransfer::push_back.cpp
//  NSolver
//
//  Created by 乔磊 on 15/10/13.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//

#include <NSolver/CellDataTransfer.h>

namespace NSFEMSolver
{
  template <int dim, typename InternalValueType>
  template <typename UserValueType>
  unsigned int
  CellDataTransfer<dim, InternalValueType>::push_back(
    UserValueType *const data_src,
    const size_type      size)
  {
    Assert(size >= tria.n_active_cells(), ExcMessage("Not enough data."));

    const size_type new_position = vector_data_ptr.size();
    const size_type active_size  = std::min(size, tria.n_active_cells());

    vector_data_ptr.push_back(std::vector<InternalValueType>(active_size));
    std::copy(data_src,
              data_src + active_size,
              vector_data_ptr[new_position].begin());

    return (new_position);
  }

  // Explicit instantiation
  template unsigned int
  CellDataTransfer<2, double>::push_back<double>(double *const   data_src,
                                                 const size_type size);
  template unsigned int
  CellDataTransfer<2, double>::push_back<float>(float *const    data_src,
                                                const size_type size);
  template unsigned int
  CellDataTransfer<3, double>::push_back<double>(double *const   data_src,
                                                 const size_type size);
  template unsigned int
  CellDataTransfer<3, double>::push_back<float>(float *const    data_src,
                                                const size_type size);
#include "CellDataTransfer.inst"
} // namespace NSFEMSolver
