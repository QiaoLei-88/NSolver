
#ifndef __MDFILU__H__
#define __MDFILU__H__

#include <deal.II/lac/sparse_matrix_ez.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/std_cxx11/array.h>
DEAL_II_DISABLE_EXTRA_DIAGNOSTICS
#include <Epetra_Operator.h>
#include <Sacado.hpp>
DEAL_II_ENABLE_EXTRA_DIAGNOSTICS
#include <deal.II/base/timer.h>

#include <fstream>

#include <NSolver/types.h>

// #define VERBOSE_OUTPUT

using namespace dealii;
using namespace NSFEMSolver;

class MDFILU : public Epetra_Operator
{
//
// private types and static variables
private:
  typedef LA::MPI::SparseMatrix SourceMatrix;
  typedef SparseMatrixEZ<double> DynamicMatrix;
  typedef unsigned short level_type;
  typedef SparseMatrixEZ<double> LevelMatrix;
  typedef unsigned short local_index_type;
  typedef unsigned long global_index_type;
  typedef double data_type;
  typedef Vector<data_type> MDFVector;
  typedef bool flag_type;

  static const global_index_type invalid_index;
  static const level_type fill_in_level_for_original_entry = 1;
  static const data_type very_large_number;

  class Indicator
  {
  public:
    data_type discarded_value;
    global_index_type n_discarded;
    global_index_type n_fill;
    global_index_type index;
    void init();
    bool operator< (const Indicator &op) const;
    bool operator== (const Indicator &op) const;
  };

  struct EntryInfo
  {
    global_index_type column;
    level_type fill_level;
    data_type value;
  };

//
// Public function interfaces
public:
  // A cheap constructor. Cheap means it doesn't allocate large amount
  // memory or do massive data copy. It just set some scalar member
  // variables. Note that the object is unusable after declared with
  // this constructor. You can initialize the object with reinit() at
  // any time before you want to apply the preconditioner.
  MDFILU (const SourceMatrix &matrix);
  MDFILU (const SourceMatrix &matrix,
          const global_index_type estimated_row_length_in,
          const global_index_type fill_in_threshold_in);

  void reinit (const global_index_type estimated_row_length_in,
               const global_index_type fill_in_threshold_in);

  global_index_type number_of_new_fill_ins() const;

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

//
// private data and function members
private:
  MPI_Comm              mpi_communicator;
  ConditionalOStream    pcout;
  TimerOutput           ILU_timer;
  // As required by Epetra_Operator,
  // all apply* functions must be const,
  // so, use timer in these functions
  // with this "const" pointer.
  TimerOutput *const    timer_ptr;

  bool metrix_factored;

  SourceMatrix const *const source_matrix;
  Epetra_CrsMatrix const *const epetra_matrix;
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

  global_index_type estimated_row_length;
  level_type fill_in_threshold;
  global_index_type n_total_fill_in;
  DynamicMatrix LU;
  // Record fill-in level for all non-zero entries, we need this to compute
  // level for new fill-ins.
  // SparseMatrixEZ<double> fill_in_level (degree,degree,degree);
  LevelMatrix fill_in_level;

  // Where is the k-th row and column in LU
  std::vector<global_index_type> permute_logical_to_storage;
  // For k-th row and column in LU, where is it in the permuted matrix
  std::vector<global_index_type> permuta_storage_to_logical;

  std::vector<Indicator> indicators;
  std::set<Indicator> sorted_indicators;

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

  // private member functions
  global_index_type get_info_of_non_zeros (
    const global_index_type row_to_factor,
    std::vector<EntryInfo> &incides_need_update,
    const bool except_pivot) const;

  void compute_discarded_value (const unsigned int row_to_factor, const bool update);

  void MDF_reordering_and_ILU_factoring();
};

// in-line member functions
inline
const std::vector<MDFILU::global_index_type> &MDFILU::get_permutation() const
{
  return (permute_logical_to_storage);
}

inline
const MDFILU::DynamicMatrix &MDFILU::get_LU() const
{
  return (LU);
}

inline
MDFILU::global_index_type MDFILU::number_of_new_fill_ins() const
{
  return (n_total_fill_in);
}

// Interfaces for Epetra_Operator

inline
double MDFILU::NormInf() const
{
  return (MDFILU::very_large_number);
}

inline
bool MDFILU::HasNormInf() const
{
  return (has_norm_infty);
}

inline
bool MDFILU::UseTranspose() const
{
  return (use_transpose);
}

inline
const char *MDFILU::Label() const
{
  return (label);
}

inline
int MDFILU::SetUseTranspose (const bool in)
{
  use_transpose = in;
  return (0);
}

inline
const Epetra_Comm &MDFILU::Comm() const
{
  return (* (epetra_comm));
}

inline
const Epetra_Map &MDFILU::OperatorDomainMap() const
{
  return (operator_domain_map);
}

inline
const Epetra_Map &MDFILU::OperatorRangeMap() const
{
  return (operator_range_map);
}

#endif
//     of #ifndef __MDFILU__H__
