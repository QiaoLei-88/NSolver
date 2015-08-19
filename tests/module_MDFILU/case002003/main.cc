#include "../MDFILU_test.h"

int main (int argc, char *argv[])
{
// #ifdef VERBOSE_OUTPUT
//   debugStream.open("debug.out");
// #endif

  Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv, ::numbers::invalid_unsigned_int);
  const global_index_type degree (10);
  const unsigned int estimated_row_length (10);
  SourceMatrix system_matrix (degree, degree, /*max_entries_per_row*/estimated_row_length);

  // Set value for system_matrix
  const bool use_sparse_matrix (true);
  if (use_sparse_matrix)
    {
      std::ifstream fin ("sparse_matrix.dat");
      while (true)
        {
          global_index_type i,j;
          data_type value;
          fin >> i >> j >> value;
          if (fin.eof())
            {
              break;
            }
          system_matrix.set (i,j,value);
        }
      fin.close();
    }
  else
    {
      std::ifstream fin ("matrix.dat");
      for (global_index_type i=0; i<degree; ++i)
        for (global_index_type j=0; j<degree; ++j)
          {
            data_type value;
            fin >> value;
            system_matrix.set (i,j,value);
          }
      fin.close();
    }

  system_matrix.compress (VectorOperation::insert);
  {
    // Out put system_matrix
    std::ofstream fout ("matrix.out");
    system_matrix.print (fout);
    fout.close();
  }
  MDFILU mdfilu (system_matrix, estimated_row_length, 2);

  // Out put LU
  {
    std::ofstream fout ("LU.out");
    mdfilu.get_LU().print (fout);
    fout.close();
  }

  // Test the MDFILU class by multiplying LU back
  // For now the sparsity pattern is ignored.
  {
    const data_type tolerance (1e-12);
    DynamicMatrix A (degree, degree, estimated_row_length);
    // Compute LD*U
    for (global_index_type i=0; i<degree; ++i)
      {
        const global_index_type i_row = mdfilu.get_permutation()[i];
        for (global_index_type j=0; j<degree; ++j)
          {
            const global_index_type j_col = mdfilu.get_permutation()[j];
            data_type value = 0;
            global_index_type vmult_max_index = 0;
            // Diagonal values of L is always 1 thus not been stored.
            // Recover its effect manually.
            if (j>=i)
              {
                value = mdfilu.get_LU().el (i_row,j_col);
                vmult_max_index = i;
              }
            else
              {
                vmult_max_index = j+1;
              }
            for (global_index_type k=0; k<vmult_max_index; ++k)
              {
                const global_index_type k_permuted = mdfilu.get_permutation()[k];
                value += mdfilu.get_LU().el (i_row,k_permuted) * mdfilu.get_LU().el (k_permuted,j_col);
              }
            if (std::abs (value) > tolerance)
              {
                A.set (i_row, j_col, value);
              }
          }
      }
    std::ofstream fout ("A.out");
    A.print (fout);
    fout.close();
  }

  // Apply LU and (LU)^-1 to a vector
  MDFVector v (degree);
  {
    const data_type a[degree] = {6, 10, 9, 5, 2, 5, 7, 3, 6, 3};

    for (global_index_type i=0; i<degree; ++i)
      {
        v[i] = a[i];
      }
  }

  // Cache initial transpose status
  const bool use_tranpose_init_stats = mdfilu.UseTranspose();
//--------- Deal.II native Vector ---------//
  mdfilu.SetUseTranspose (false);
  {
    MDFVector o (v);
    mdfilu.apply (o,o);
    std::ofstream fout ("apply.out");
    fout << "Vector v:" << std::endl;
    fout << v;
    fout << "Vector (LU)*v:" << std::endl;
    fout << o << std::endl;
    fout.close();
  }
  {
    MDFVector o (v);
    mdfilu.apply_inverse (o,o);
    std::ofstream fout ("apply.out", std::fstream::app);
    fout << "Vector v:" << std::endl;
    fout << v;
    fout << "Vector ((LU)^-1)*v:" << std::endl;
    fout << o << std::endl;
    fout.close();
  }

  mdfilu.SetUseTranspose (true);
  {
    MDFVector o (v);
    mdfilu.apply (o,o);
    std::ofstream fout ("apply.out", std::fstream::app);
    fout << "Vector v:" << std::endl;
    fout << v;
    fout << "Vector ((LU)^T)*v:" << std::endl;
    fout << o << std::endl;
    fout.close();
  }
  {
    MDFVector o (v);
    mdfilu.apply_inverse (o,o);
    std::ofstream fout ("apply.out", std::fstream::app);
    fout << "Vector v:" << std::endl;
    fout << v;
    fout << "Vector (((LU)^-1)^T)*v:" << std::endl;
    fout << o << std::endl;
    fout.close();
  }
  // Set flags
  mdfilu.SetUseTranspose (use_tranpose_init_stats);
//[END]// //--------- Deal.II native Vector ---------//

//--------- Finally, Preconditioner test ---------//
  {
    // No preconditioner first
    Epetra_Vector o (system_matrix.trilinos_matrix().DomainMap());
    Epetra_Vector b (View, system_matrix.trilinos_matrix().RangeMap(),
                     v.begin());
    AztecOO solver;
    solver.SetAztecOption (AZ_output, AZ_none);
    solver.SetAztecOption (AZ_solver, AZ_gmres);
    solver.SetLHS (&o);
    solver.SetRHS (&b);

    solver.SetAztecOption (AZ_precond,         AZ_none);
    solver.SetUserMatrix (const_cast<Epetra_CrsMatrix *>
                          (&system_matrix.trilinos_matrix()));

    const unsigned int max_iterations = 1000;
    const double  linear_residual = 1e-8;
    solver.Iterate (max_iterations, linear_residual);

    std::ofstream fout ("apply.out", std::fstream::app);
    fout << "GMRES Solved Ax=v without preconditioner:" << std::endl;
    fout << "NumIters: " << solver.NumIters() << std::endl;
    fout << "TrueResidual: " << solver.TrueResidual() << std::endl;
    fout.precision (3);
    fout << std::scientific;
    for (global_index_type i=0; i<degree; ++i)
      {
        fout << o[i] << ' ';
      }
    fout << std::endl << std::endl;
    fout.close();
  }
  {
    // No preconditioner first
    Epetra_Vector o (system_matrix.trilinos_matrix().DomainMap());
    Epetra_Vector b (View, system_matrix.trilinos_matrix().RangeMap(),
                     v.begin());
    AztecOO solver;
    solver.SetAztecOption (AZ_output, AZ_none);
    solver.SetAztecOption (AZ_solver, AZ_gmres);
    solver.SetLHS (&o);
    solver.SetRHS (&b);

    solver.SetUserMatrix (const_cast<Epetra_CrsMatrix *>
                          (&system_matrix.trilinos_matrix()));

    mdfilu.SetUseTranspose (false);
    solver.SetPrecOperator (&mdfilu);

    const unsigned int max_iterations = 1000;
    const double  linear_residual = 1e-8;
    solver.Iterate (max_iterations, linear_residual);

    std::ofstream fout ("apply.out", std::fstream::app);
    fout << "GMRES Solved Ax=v with MDFILU preconditioner:" << std::endl;
    fout << "NumIters: " << solver.NumIters() << std::endl;
    fout << "TrueResidual: " << solver.TrueResidual() << std::endl;
    fout.precision (3);
    fout << std::scientific;
    for (global_index_type i=0; i<degree; ++i)
      {
        fout << o[i] << ' ';
      }
    fout << std::endl << std::endl;
    fout.close();
  }
  mdfilu.SetUseTranspose (use_tranpose_init_stats);
//[END]// //--------- Finally, Preconditioner test ---------//

// #ifdef VERBOSE_OUTPUT
//   debugStream.close();
// #endif
  return (system("cat  A.out  apply.out  LU.out  matrix.out  > all.out"));
}
