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

  template<int dim, typename Number>
  template <typename ValueType>
  unsigned int
  CellDataTransfer<dim, Number>::push_back (ValueType *const data_src,
                                            const size_type  size)
  {
    Assert (size <= active_data_size,
            ExcMessage ("Not enough data."));

    const size_type new_position = vector_data_ptr.size();
    // Allocate 30% more memory for future using.
    vector_data_ptr.resize (new_position + 1, std::vector<Number> (size * 1.3));
    std::copy (data_src, data_src+size, vector_data_ptr[new_position].begin());

    return (new_position);
  }

// Explicit instantiation
  template
  unsigned int
  CellDataTransfer<2, double>::push_back<double> (double *const   data_src,
                                                  const size_type size);
  template
  unsigned int
  CellDataTransfer<2, double>::push_back<float> (float *const    data_src,
                                                 const size_type size);
  template
  unsigned int
  CellDataTransfer<3, double>::push_back<double> (double *const   data_src,
                                                  const size_type size);
  template
  unsigned int
  CellDataTransfer<3, double>::push_back<float> (float *const    data_src,
                                                 const size_type size);
#include "CellDataTransfer.inst"
}
