
#ifndef __MDFILU__H__
#define __MDFILU__H__

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/sparse_matrix_ez.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/std_cxx11/array.h>
DEAL_II_DISABLE_EXTRA_DIAGNOSTICS
#include <Epetra_Operator.h>
#include <Sacado.hpp>
DEAL_II_ENABLE_EXTRA_DIAGNOSTICS

#include <fstream>

#define VERBOSE_OUTPUT

using namespace dealii;

#define USE_TRILINOS_LA
namespace LA
{
#ifdef USE_PETSC_LA
  using namespace dealii::LinearAlgebraPETSc;
#else
  using namespace dealii::LinearAlgebraTrilinos;
#endif
}

#ifndef __NSVector__DEFINED__
typedef LA::MPI::Vector NSVector;
#define __NSVector__DEFINED__
#endif
typedef LA::MPI::SparseMatrix SourceMatrix;
typedef SparseMatrixEZ<double> DynamicMatrix;
typedef unsigned short local_index_type;
typedef unsigned long global_index_type;
typedef double data_type;
typedef Vector<data_type> MDFVector;
typedef bool flag_type;

class MDFILU : public Epetra_Operator
{
public:
  MDFILU (const SourceMatrix &matrix,
          const global_index_type estimated_row_length_in,
          const global_index_type fill_in_threshold_in);

  int apply (const data_type *const in, data_type *const out) const;
  int apply_transpose (const data_type *const in, data_type *const out) const;

  int apply_inverse (const data_type *const in, data_type *const out) const;
  int apply_inverse_transpose (const data_type *const in, data_type *const out) const;

  // Dedicated interface to dealii Vector<double>
  int apply (const Vector<data_type> &in, Vector<data_type> &out) const;
  int apply_inverse (const Vector<data_type> &in, Vector<data_type> &out) const;

  int apply (const NSVector &in, NSVector &out) const;
  int apply_inverse (const NSVector &in, NSVector &out) const;

  // Interface for debug code
  const std::vector<global_index_type> &get_permutation() const;
  const DynamicMatrix &get_LU() const;
  ~MDFILU();

  // Virtual functions from Epetra_Operator

  virtual int Apply (const Epetra_MultiVector &, Epetra_MultiVector &) const;
  virtual int ApplyInverse (const Epetra_MultiVector &, Epetra_MultiVector &) const;
  virtual double NormInf() const;
  virtual bool UseTranspose() const;
  virtual bool HasNormInf() const;

  virtual const char *Label() const;

  virtual const Epetra_Comm &Comm() const;
  virtual const Epetra_Map &OperatorDomainMap() const;
  virtual const Epetra_Map &OperatorRangeMap() const;

  virtual int SetUseTranspose (const bool);

private:
  const global_index_type invalid_index;
  const data_type very_large_number;

#define N_INDICATOR 3
  class Indicator: public std_cxx11::array<data_type,N_INDICATOR>
  {
  public:
    void init();
    int operator- (const Indicator &op) const;
  };

  void get_indices_of_non_zeros (
    const global_index_type row_to_factor,
    std::vector<global_index_type> &incides_need_update,
    const bool except_pivot) const;

  void compute_discarded_value (const unsigned int row_to_factor);

  global_index_type find_min_discarded_value() const;
  void MDF_reordering_and_ILU_factoring();


  const global_index_type degree;

  // Because the Sparse matrix usually returns a zero when element (i,j)
  // is out of the sparsity pattern. So it is not convenient to use zero
  // as fill-in level for original entries.
  // Here I set the fill-in level with offset 1.
  // This is to say when fill-in level equals
  //    0  :  level infinite, new fill-in
  //    1  :  level 0 in article, original entry
  //    2  :  level 1 fill in
  //    ... so on the same.
  static const global_index_type fill_in_level_for_original_entry = 1;
  const global_index_type estimated_row_length;
  const global_index_type fill_in_threshold;
  DynamicMatrix LU;
  // Record fill-in level for all non-zero entries, we need this to compute
  // level for new fill-ins.
  // SparseMatrixEZ<double> fill_in_level (degree,degree,degree);
  DynamicMatrix fill_in_level;

  // Where is the k-th row and column in LU
  std::vector<global_index_type> permute_logical_to_storage;
  // For k-th row and column in LU, where is it in the permuted matrix
  std::vector<global_index_type> permuta_storage_to_logical;

  std::vector<Indicator> indicators;

  // During factoring procedure, we need to go through all un-factored entries that connected
  // with this row, i.e., for all k that a(i_row, k) \ne 0 and a(k, i_row) \ne 0.
  // That's why we need the flag array row_factored.
  std::vector<flag_type> row_factored;


  // Data for Epetra_Operator interface
  bool use_transpose;
  const bool has_norm_infty;
  const Epetra_Comm *epetra_comm;
  const Epetra_Map operator_domain_map;
  const Epetra_Map operator_range_map;

  const static char label[];
};

#endif
//     of #ifndef __MDFILU__H__
