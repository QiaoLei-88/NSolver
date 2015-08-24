
#ifndef __FEMNSolver__Tools__
#define __FEMNSolver__Tools__

#include <NSolver/types.h>

namespace NSFEMSolver
{
  using namespace dealii;
  namespace Tools
  {
    template<typename Matrix>
    void write_matrix_MTX (std::ostream &out, const Matrix &matrix);
  }
}
#endif
