//
//  CellDataTransfer::CellDataTransfer.cpp
//  NSolver
//
//  Created by 乔磊 on 15/10/13.
//  Copyright (c) 2015年 乔磊. All rights reserved.
//

#include <NSolver/CellDataTransfer.h>

namespace NSFEMSolver
{

  template<int dim, typename InternalValueType>
  CellDataTransfer<dim, InternalValueType>::CellDataTransfer (Triangulation<dim> &tria_in)
    :
    active_data_size (tria_in.n_active_cells()),
    tria (tria_in)
  {
    Assert (tria_in.n_vertices() != 0,
            ExcMessage ("Do not attach empty triangulation to CellDataTransfer object."));

    // Initial data is assumed to have the same order as active cells
    const typename Triangulation<dim>::active_cell_iterator
    endc = tria_in.end();
    typename Triangulation<dim>::active_cell_iterator
    cell = tria_in.begin_active();
    for (; cell != endc; ++cell)
      {
        cell->set_user_index (cell->active_cell_index());
      }

    tria_in.signals.pre_coarsening_on_cell.connect
    (std_cxx11::bind (&CellDataTransfer<dim, InternalValueType>::children_to_parent,
                      this,
                      std_cxx11::placeholders::_1));

    tria_in.signals.post_refinement_on_cell.connect
    (std_cxx11::bind (&CellDataTransfer<dim, InternalValueType>::parent_to_children,
                      this,
                      std_cxx11::placeholders::_1));
  }


  template<int dim, typename InternalValueType>
  CellDataTransfer<dim, InternalValueType>::~CellDataTransfer()
  {
    clear();
  }

#include "CellDataTransfer.inst"
}
