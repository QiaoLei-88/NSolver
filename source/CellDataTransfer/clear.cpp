//
//  CellDataTransfer::clear.cpp
//  NSolver
//
//  Created by 乔磊 on 15/10/13.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//

#include <NSolver/CellDataTransfer.h>

namespace NSFEMSolver
{

  template<int dim, typename Number>
  void CellDataTransfer<dim, Number>::clear()
  {
    for (unsigned int i=0; i<vector_data_ptr.size(); ++i)
      {
        vector_data_ptr[i].clear();
      }
    vector_data_ptr.clear();

    return;
  }

#include "CellDataTransfer.inst"
}
