
#include <NSolver/Tools.h>
#include <deal.II/lac/sparse_matrix_ez.h>
#include <NSolver/MDFILU/MapMatrix.h>


namespace NSFEMSolver
{
  using namespace dealii;
  namespace Tools
  {

    template<typename Matrix>
    void write_matrix_MTX (std::ostream &out, const Matrix &matrix)
    {
      AssertThrow (out, ExcIO());

      out <<"%%MatrixMarket matrix coordinate real general\n";
      out << matrix.m() << "\t" << matrix.n() << "\t"
          << matrix.n_nonzero_elements() << std::endl;

      for (typename Matrix::size_type row=0; row<matrix.m(); ++row)
        {
          const typename Matrix::size_type row_out = row + 1;
          typename Matrix::const_iterator iter = matrix.begin (row);
          const typename Matrix::const_iterator end_iter = matrix.end (row);
          for (; iter != end_iter; ++iter)
            {
              out << row_out << '\t'
                  << iter->column() + 1 << '\t'
                  << iter->value() << std::endl;
            }
        }
      return;
    }


    template
    void write_matrix_MTX<NSMatrix> (std::ostream &out, const NSMatrix &matrix);
    template
    void write_matrix_MTX<SparseMatrixEZ<double> > (std::ostream &out, const SparseMatrixEZ<double>  &matrix);
    template
    void write_matrix_MTX<MapMatrix<double> > (std::ostream &out, const MapMatrix<double> &matrix);
  }
}
