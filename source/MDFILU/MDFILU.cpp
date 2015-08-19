
#include <NSolver/MDFILU/MDFILU.h>

const char MDFILU::label[] = "Epetra_Operator_MDFILU";
const MDFILU::data_type MDFILU::very_large_number
  = std::numeric_limits<MDFILU::data_type>::max();
const MDFILU::global_index_type MDFILU::invalid_index
  = std::numeric_limits<MDFILU::global_index_type>::max();

MDFILU::MDFILU (const SourceMatrix &matrix,
                const global_index_type estimated_row_length_in,
                const global_index_type fill_in_threshold_in)
  :
  mpi_communicator (MPI_COMM_WORLD),
  pcout (std::cout,
         (Utilities::MPI::this_mpi_process (mpi_communicator) == 0)),
  ILU_timer (MPI_COMM_WORLD,
             pcout,
             TimerOutput::never,
             TimerOutput::wall_times),
  degree (matrix.m()),
  estimated_row_length (estimated_row_length_in),
  fill_in_threshold (fill_in_threshold_in + fill_in_level_for_original_entry),
  LU (degree, degree, estimated_row_length),
  fill_in_level (degree, degree, estimated_row_length),
  permute_logical_to_storage (degree, MDFILU::invalid_index),
  permuta_storage_to_logical (degree, MDFILU::invalid_index),
  indicators (degree),
  row_factored (degree, false),
  use_transpose (false),
  has_norm_infty (false),
  epetra_comm (& (matrix.trilinos_matrix().Comm())),
  operator_domain_map (matrix.trilinos_matrix().DomainMap()),
  operator_range_map (matrix.trilinos_matrix().RangeMap())
{
  //initialize the LU matrix
  LU.copy_from (matrix, /*elide_zero_values=*/false);
  MDF_reordering_and_ILU_factoring();
}

MDFILU::~MDFILU()
{
  row_factored.clear();
  indicators.clear();
  permute_logical_to_storage.clear();
  fill_in_level.clear();
  LU.clear();
}

void MDFILU::Indicator::init()
{
  discarded_value = 0.0;
  n_discarded     = 0;
  n_fill          = 0;
  return;
}

bool MDFILU::Indicator::operator< (const Indicator &op) const
{
  if (discarded_value != op.discarded_value)
    {
      return (discarded_value < op.discarded_value);
    }

  if (n_discarded     != op.n_discarded)
    {
      return (n_discarded < op.n_discarded);
    }

  // We must make decision on the last element
  return (n_fill < op.n_fill);
}

MDFILU::global_index_type MDFILU::get_info_of_non_zeros (
  const global_index_type row_to_factor,
  std::vector<EntryInfo> &incides_need_update,
  const bool except_pivot) const
{
  global_index_type n_non_zero = 0;
  typename DynamicMatrix::const_iterator iter_fill = fill_in_level.begin (row_to_factor);
  const DynamicMatrix::const_iterator end_fill = fill_in_level.end (row_to_factor);

  typename DynamicMatrix::const_iterator iter_col = LU.begin (row_to_factor);
  const DynamicMatrix::const_iterator end_col = LU.end (row_to_factor);

  (void)end_fill;
  Assert (end_fill == end_col,
          ExcMessage ("Sparsity pattern of LU and fill_level mismatch!"));

  for (; iter_col < end_col; ++iter_col, ++iter_fill)
    {
      Assert (iter_fill == iter_col,
              ExcMessage ("Sparsity pattern of LU and fill_level mismatch!"));
      const global_index_type j_col = iter_col->column();
      if (j_col == row_to_factor && except_pivot)
        {
          // If we do not want to count on pivot, jump over
          continue;
        }

      if (!row_factored[j_col])
        {
          const global_index_type vector_size = incides_need_update.size();
          if (vector_size <= n_non_zero)
            {
              incides_need_update.resize (vector_size + estimated_row_length);
            }
          incides_need_update[n_non_zero].column = j_col;
          incides_need_update[n_non_zero].value = iter_col->value();
          incides_need_update[n_non_zero].fill_level =
            static_cast<level_type> (iter_fill->value());
          ++n_non_zero;
        }
    }
  return (n_non_zero);
}


void MDFILU::compute_discarded_value (const unsigned int row_to_factor)
{
  Indicator &return_value = indicators[row_to_factor];
#ifdef VERBOSE_OUTPUT
  if (row_to_factor == 0)
    {
      std::ofstream f_level ("fill_level_cdv.out");
      fill_in_level.print (f_level);
      f_level.close();
    }
#endif

  return_value.init();
  const data_type pivot = LU.el (row_to_factor, row_to_factor);

  if (pivot==0.0)
    {
      return_value.n_fill          = MDFILU::invalid_index;
      return_value.n_discarded     = MDFILU::invalid_index;
      return_value.discarded_value = MDFILU::very_large_number;
      return;
    }

  // compute discarded value for i_row := row_to_factor.
  // During this procedure, we need to go through all un-factored entries that connected
  // with this row, i.e., for all k that a(i_row, k) \ne 0 and a(k, i_row) \ne 0.
  // That's why we need the flag array row_factored.

  // Find number of rows need to go through. The value is all non-zero
  // entries except the pivot and factored rows.
  std::vector<EntryInfo> incides_need_update (estimated_row_length);
  const bool except_pivot (true);
  const global_index_type n_row_need_update =
    get_info_of_non_zeros (row_to_factor, incides_need_update, except_pivot);

  const data_type pivot_inv = 1.0/pivot;

  for (global_index_type i=0; i<n_row_need_update; ++i)
    {
      const global_index_type i_row = incides_need_update[i].column;
      const data_type value_of_row_pivot = LU.el (i_row, row_to_factor) * pivot_inv;
      const level_type fill_level_of_row_pivot = fill_in_level.el (i_row,row_to_factor);
      for (global_index_type j=0; j<n_row_need_update; ++j)
        {
          const global_index_type j_col = incides_need_update[j].column;
          // Check fill-in level
          data_type new_fill_in_level = fill_in_level.el (i_row, j_col);
          if (new_fill_in_level == 0 /* fill in level for new entry*/)
            {
              ++ return_value.n_fill;

              // Make sure that the provided fill_in_threshold consists with
              // the internal definition, i.e., has an offset.
              new_fill_in_level =
                (incides_need_update[j].fill_level - fill_in_level_for_original_entry) +
                (fill_level_of_row_pivot - fill_in_level_for_original_entry) +
                1 +
                fill_in_level_for_original_entry;
            }
          if (new_fill_in_level > fill_in_threshold
              &&
              i_row != j_col) //Never drop diagonal element
            {
              // Element will be discarded
              const data_type update = incides_need_update[j].value * value_of_row_pivot;
              return_value.discarded_value += update*update;
              ++ return_value.n_discarded;
            }
        } // For each column need update
    } // For each row need update

  return;
}

// Determine the next row to be factored by finding out the one with minimum
// indicator form rows that have not been factored.
MDFILU::global_index_type MDFILU::find_min_discarded_value() const
{
  global_index_type candidate (0);
  bool need_init_candidate (true);
  for (global_index_type i=0; i<indicators.size(); ++i)
    {
      if (row_factored[i])
        {
          continue;
        }
      // Set first un-factored row as candidate if it is not initialized
      if (need_init_candidate)
        {
          candidate = i;
          need_init_candidate = false;
        }

      if (indicators[i] < indicators[candidate])
        {
          candidate = i;
        }
    }
  return (candidate);
}


void MDFILU::MDF_reordering_and_ILU_factoring()
{
#ifdef VERBOSE_OUTPUT
  std::ofstream debugStream ("debug.out");
#endif

  // Initialize::BEGIN
  // Compute Initial fill in level
  ILU_timer.reset();
  ILU_timer.enter_subsection ("Compute initial fill in level");
  for (global_index_type i_row=0; i_row<degree; ++i_row)
    {
      for (typename DynamicMatrix::const_iterator iter_col = LU.begin (i_row);
           iter_col < LU.end (i_row); ++iter_col)
        {
          fill_in_level.set (i_row,
                             iter_col->column(),
                             fill_in_level_for_original_entry);
          // // a(i,j) exists?
          // if (system_matrix. (i,j) == 0.0)
          //   {
          //     fill_in_level.add (i, j, numbers::invalid_unsigned_int);
          //   }
          // else
          //   {
          //     fill_in_level.add (i, j, 0);
          //   }
        }
    }
  ILU_timer.leave_subsection ("Compute initial fill in level");
  // Compute initial discarded value, must be done after all fill-in level have
  // been set.
  ILU_timer.enter_subsection ("Compute initial discarded value");
  for (global_index_type i_row=0; i_row<degree; ++i_row)
    {
      compute_discarded_value (i_row);
    }
  ILU_timer.leave_subsection ("Compute initial discarded value");
  // Initialize::END
#ifdef VERBOSE_OUTPUT
  {
    std::ofstream f_level ("fill_level.out");
    fill_in_level.print (f_level);
    f_level.close();
  }
#endif

  // Factoring the matrix
  for (global_index_type n_row_factored=0; n_row_factored<degree; ++n_row_factored)
    {
      // Find the row with minimal discarded value
      ILU_timer.enter_subsection ("Find min discarded value");
      const global_index_type row_to_factor
        = find_min_discarded_value();
      ILU_timer.leave_subsection ("Find min discarded value");
#ifdef VERBOSE_OUTPUT
      for (global_index_type i=0; i<degree; ++i)
        {
          debugStream << indicators.at (i).at (0) << "\t ";
        }
      debugStream << std::endl;
      for (global_index_type i=0; i<degree; ++i)
        {
          debugStream << indicators.at (i).at (1) << "\t ";
        }
      debugStream << std::endl;
      for (global_index_type i=0; i<degree; ++i)
        {
          debugStream << indicators.at (i).at (2) << "\t ";
        }
      debugStream << std::endl;
#endif
      ILU_timer.enter_subsection ("Prepare factorization");
      row_factored[row_to_factor] = true;
      permute_logical_to_storage[n_row_factored] = row_to_factor;
      permuta_storage_to_logical[row_to_factor] = n_row_factored;

#ifdef VERBOSE_OUTPUT
      debugStream << "row_to_factor: " << row_to_factor << std::endl;
#endif

      // Find number of rows need to go through. The value is all non-zero
      // entries except the pivot and factored rows.

      const data_type pivot = LU.diagonal (row_to_factor)->value();
      const data_type pivot_inv = 1.0/pivot;

      const bool except_pivot (true);
      std::vector<EntryInfo> incides_need_update (estimated_row_length);

      const global_index_type n_row_need_update =
        get_info_of_non_zeros (row_to_factor, incides_need_update, except_pivot);

      ILU_timer.leave_subsection ("Prepare factorization");
      ILU_timer.enter_subsection ("Factorization:");
      for (global_index_type i=0; i<n_row_need_update; ++i)
        {
          const global_index_type i_row = incides_need_update[i].column;
          // Update current column, i.e., lower triangle part
          // Doing this via iterator is more efficient, but now there is no
          // non-const iterator available.

          const data_type value_of_row_pivot = LU.el (i_row, row_to_factor) * pivot_inv;
          LU.set (i_row,row_to_factor, value_of_row_pivot, /*elide_zero_values=*/ false);

          const level_type fill_level_of_row_pivot = fill_in_level.el (i_row,row_to_factor);
          // Update the remaining matrix
          for (global_index_type j=0; j<n_row_need_update; ++j)
            {
              const global_index_type j_col = incides_need_update[j].column;
              // Check fill-in level
              unsigned int new_fill_in_level
                = static_cast<level_type> (fill_in_level.el (i_row, j_col));
              if (new_fill_in_level == 0 /* fill in level for new entry*/)
                {
                  new_fill_in_level =
                    (incides_need_update[j].fill_level - fill_in_level_for_original_entry) +
                    (fill_level_of_row_pivot - fill_in_level_for_original_entry) +
                    1 +
                    fill_in_level_for_original_entry;
                }

              // Make sure that the provided fill_in_threshold consists with
              // the internal definition, i.e., has a offset one. See documentation
              // above for details
              if (new_fill_in_level <= fill_in_threshold
                  ||
                  i_row == j_col) //Always keep diagonal element)
                {
                  // Element accepted
                  const data_type value = LU.el (i_row, j_col);
                  const data_type update = value - incides_need_update[j].value * value_of_row_pivot;

                  // Have no information of the existence of this value.
                  // A search operation implied.
                  LU.set (i_row, j_col, update, /*elide_zero_values=*/ false);
                  // Update fill-level if this is a new entry
                  fill_in_level.set (i_row, j_col, new_fill_in_level);
                }
            } // For each column need update
        } // For each row need update
      ILU_timer.leave_subsection ("Factorization:");
      ILU_timer.enter_subsection ("Update discarded value");
      for (global_index_type i=0; i<n_row_need_update; ++i)
        {
          const global_index_type i_row = incides_need_update[i].column;
          compute_discarded_value (i_row);
        }
      ILU_timer.leave_subsection ("Update discarded value");
    } // For each row in matrix
  ILU_timer.print_summary();
  ILU_timer.reset();
  return;
}


// This function is safe even @p in and @out is the same vector.
// Because we only multiply the vector with upper and lower triangle
// matrix in order, the passed vector value is never used again.
int MDFILU::apply (const data_type *const in, data_type *const out) const
{
  // Apply U to in
  for (global_index_type i=0; i<degree; ++i)
    {
      // Forward sweep
      const global_index_type i_row = permute_logical_to_storage[i];

      data_type value = 0;
      for (typename DynamicMatrix::const_iterator iter_col = LU.begin (i_row);
           iter_col < LU.end (i_row); ++iter_col)
        {
          const global_index_type j_col = iter_col->column();
          const global_index_type j = permuta_storage_to_logical[j_col];
          if (j >= i)
            {
              // Upper triangle only
              value += iter_col->value() * in[j_col];
            }
        }
      out[i_row] = value;
    }

  // Apply L to the result of U*in
  for (global_index_type ii=degree; ii>0; --ii)
    {
      // backward sweep; be careful on "ii-1" because ii is unsigned
      const global_index_type i = ii - 1;
      const global_index_type i_row = permute_logical_to_storage[i];

      // Diagonal value of L is alway 1, so we can just accumulate on out[i].
      for (typename DynamicMatrix::const_iterator iter_col = LU.begin (i_row);
           iter_col < LU.end (i_row); ++iter_col)
        {
          const global_index_type j_col = iter_col->column();
          const global_index_type j = permuta_storage_to_logical[j_col];
          if (j < i)
            {
              // Lower triangle only
              out[i_row] += iter_col->value() * out[j_col];
            }
        }
    }

  return (0);
}

int MDFILU::apply_transpose (const data_type *const in, data_type *const out) const
{

  // Apply L^T to in
  for (global_index_type i=0; i<degree; ++i)
    {
      // Forward sweep
      const global_index_type i_row = permute_logical_to_storage[i];

      // Diagonal value of L is alway 1, so we can just accumulate on out[i].
      for (typename DynamicMatrix::const_iterator iter_col = LU.begin (i_row);
           iter_col < LU.end (i_row); ++iter_col)
        {
          const global_index_type j_col = iter_col->column();
          const global_index_type j = permuta_storage_to_logical[j_col];
          if (j < i)
            {
              // Lower triangle only
              // Because the i-loop goes up from i=0, all j<i must have experienced
              // j==i in previous i-loop, with the assumption that diagonal entry
              // exists in all rows. So it is save to use += here.
              out[j_col] += iter_col->value() * in[i_row];
            }
          else if (j == i)
            {
              out[i_row] = in[i_row];
            }
        }
    }

  // Apply U^T to the result of (L^T)*in
  for (global_index_type ii=degree; ii>0; --ii)
    {
      // backward sweep; be careful on "ii-1" because ii is unsigned
      const global_index_type i = ii - 1;
      const global_index_type i_row = permute_logical_to_storage[i];

      data_type pivot = 0;
      for (typename DynamicMatrix::const_iterator iter_col = LU.begin (i_row);
           iter_col < LU.end (i_row); ++iter_col)
        {
          const global_index_type j_col = iter_col->column();
          const global_index_type j = permuta_storage_to_logical[j_col];
          if (j > i)
            {
              // Upper triangle only
              // Because the i-loop goes down from top, all j>i must have experienced
              // j==i in previous i-loop, with the assumption that diagonal entry
              // exists in all rows. So it is save to use += here.
              out[j_col] += iter_col->value() * out[i_row];
            }
          else if (j == i)
            {
              // We cannot overwrite out[i_row] immediately because it is used above
              // and our matrix is permuted. We don't know which if branch comes first.
              pivot = iter_col->value();
            }
        }
      Assert (pivot != 0.0, ExcMessage ("Zero pivot encountered!"));
      out[i_row] *= pivot;
    }

  return (0);
}

int MDFILU::apply_inverse_transpose (const data_type *const in, data_type *const out) const
{
  // Apply (U^T)^-1 to in
  for (global_index_type i=0; i<degree; ++i)
    {
      // Forward substitution
      const global_index_type i_row = permute_logical_to_storage[i];

      // Update vector value of current row for using below
      for (typename DynamicMatrix::const_iterator iter_col = LU.begin (i_row);
           iter_col < LU.end (i_row); ++iter_col)
        {
          const global_index_type j_col = iter_col->column();
          const global_index_type j = permuta_storage_to_logical[j_col];
          if (j == i)
            {
              Assert (iter_col->value() != 0.0,
                      ExcMessage ("Zero pivot encountered!"));
              out[i_row] = in[i_row]/iter_col->value();
              break;
            }
        }

      for (typename DynamicMatrix::const_iterator iter_col = LU.begin (i_row);
           iter_col < LU.end (i_row); ++iter_col)
        {
          const global_index_type j_col = iter_col->column();
          const global_index_type j = permuta_storage_to_logical[j_col];
          if (j > i)
            {
              // Lower triangle only
              out[j_col] -= iter_col->value() * out[i_row];
            }
        }
    }

  // Apply (L^T)^-1 to the result of ((U^T)^-1)*in
  for (global_index_type ii=degree; ii>0; --ii)
    {
      // Backward substitution; be careful on "ii-1" because ii is unsigned
      const global_index_type i = ii - 1;
      const global_index_type i_row = permute_logical_to_storage[i];

      for (typename DynamicMatrix::const_iterator iter_col = LU.begin (i_row);
           iter_col < LU.end (i_row); ++iter_col)
        {
          const global_index_type j_col = iter_col->column();
          const global_index_type j = permuta_storage_to_logical[j_col];
          if (j < i)
            {
              // Upper triangle only
              out[j_col] -= iter_col->value() * out[i_row];
            }
        }
    }

  return (0);
}

int MDFILU::apply_inverse (const data_type *const in, data_type *const out) const
{
  // Apply L^-1 to in
  for (global_index_type i=0; i<degree; ++i)
    {
      // Forward substitution
      const global_index_type i_row = permute_logical_to_storage[i];

      // Diagonal value of L is alway 1, so we can just accumulate on out[i_row].
      out[i_row] = in[i_row];

      for (typename DynamicMatrix::const_iterator iter_col = LU.begin (i_row);
           iter_col < LU.end (i_row); ++iter_col)
        {
          const global_index_type j_col = iter_col->column();
          const global_index_type j = permuta_storage_to_logical[j_col];
          if (j < i)
            {
              // Upper triangle only
              out[i_row] -= iter_col->value() * out[j_col];
            }
        }
    }

  // Apply U^-1 to the result of U*in
  for (global_index_type ii=degree; ii>0; --ii)
    {
      // Backward substitution; be careful on "ii-1" because ii is unsigned
      const global_index_type i = ii - 1;
      const global_index_type i_row = permute_logical_to_storage[i];

      data_type pivot = 0.0;

      for (typename DynamicMatrix::const_iterator iter_col = LU.begin (i_row);
           iter_col < LU.end (i_row); ++iter_col)
        {
          const global_index_type j_col = iter_col->column();
          const global_index_type j = permuta_storage_to_logical[j_col];
          if (j > i)
            {
              // Lower triangle only
              out[i_row] -= iter_col->value() * out[j_col];
            }
          else if (j == i)
            {
              pivot = iter_col->value();
            }
        }
      Assert (pivot != 0.0, ExcMessage ("Zero pivot encountered!"));
      out[i_row] /= pivot;
    }

  return (0);
}

int MDFILU::apply (const Vector<data_type> &in, Vector<data_type> &out) const
{
  Assert (in.size() == out.size(),
          ExcDimensionMismatch (in.size(), out.size()));
  Assert (in.size() == degree,
          ExcDimensionMismatch (in.size(), degree));

  if (use_transpose)
    {
      return (apply_transpose (in.begin(), out.begin()));
    }
  else
    {
      return (apply (in.begin(), out.begin()));
    }
}

int MDFILU::apply_inverse (const Vector<data_type> &in, Vector<data_type> &out) const
{
  Assert (in.size() == out.size(),
          ExcDimensionMismatch (in.size(), out.size()));
  Assert (in.size() == degree,
          ExcDimensionMismatch (in.size(), degree));

  if (use_transpose)
    {
      return (apply_inverse_transpose (in.begin(), out.begin()));
    }
  else
    {
      return (apply_inverse (in.begin(), out.begin()));
    }
}

//------------------------------//------------------------------
// Interface to deal.II Trilinos vector wrapper
int MDFILU::apply (const NSVector &in, NSVector &out) const
{
  Assert (in.size() == out.size(),
          ExcDimensionMismatch (in.size(), out.size()));
  Assert (in.size() == degree,
          ExcDimensionMismatch (in.size(), degree));

  return (Apply (in.trilinos_vector(), out.trilinos_vector()));
}

int MDFILU::apply_inverse (const NSVector &in, NSVector &out) const
{
  Assert (in.size() == out.size(),
          ExcDimensionMismatch (in.size(), out.size()));
  Assert (in.size() == degree,
          ExcDimensionMismatch (in.size(), degree));
  return (ApplyInverse (in.trilinos_vector(), out.trilinos_vector()));
}
//------------------------------//------------------------------

const std::vector<MDFILU::global_index_type> &MDFILU::get_permutation() const
{
  return (permute_logical_to_storage);
}
const MDFILU::DynamicMatrix &MDFILU::get_LU() const
{
  return (LU);
}

// Virtual functions from Epetra_Operator

int MDFILU::Apply (const Epetra_MultiVector &in, Epetra_MultiVector &out) const
{
  Assert (in.NumVectors() == out.NumVectors(),
          ExcDimensionMismatch (in.NumVectors(), out.NumVectors()));
  const global_index_type n_vectors = in.NumVectors();

  if (use_transpose)
    {
      for (global_index_type i=0; i<n_vectors; ++i)
        {
          apply_transpose (in[i], out[i]);
        }
    }
  else
    {
      for (global_index_type i=0; i<n_vectors; ++i)
        {
          apply (in[i], out[i]);
        }
    }

  return (0);
}


int MDFILU::ApplyInverse (const Epetra_MultiVector &in, Epetra_MultiVector &out) const
{
  Assert (in.NumVectors() == out.NumVectors(),
          ExcDimensionMismatch (in.NumVectors(), out.NumVectors()));

  const global_index_type n_vectors = in.NumVectors();

  if (use_transpose)
    {
      for (global_index_type i=0; i<n_vectors; ++i)
        {
          apply_inverse_transpose (in[i], out[i]);
        }
    }
  else
    {
      for (global_index_type i=0; i<n_vectors; ++i)
        {
          apply_inverse (in[i], out[i]);
        }
    }

  return (0);
}

double MDFILU::NormInf() const
{
  return (MDFILU::very_large_number);
}

bool MDFILU::HasNormInf() const
{
  return (has_norm_infty);
}

bool MDFILU::UseTranspose() const
{
  return (use_transpose);
}

const char *MDFILU::Label() const
{
  return (label);
}

int MDFILU::SetUseTranspose (const bool in)
{
  use_transpose = in;
  return (0);
}

const Epetra_Comm &MDFILU::Comm() const
{
  return (* (epetra_comm));
}

const Epetra_Map &MDFILU::OperatorDomainMap() const
{
  return (operator_domain_map);
}

const Epetra_Map &MDFILU::OperatorRangeMap() const
{
  return (operator_range_map);
}
