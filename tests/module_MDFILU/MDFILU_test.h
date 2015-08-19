#ifndef __MDFILU_TEST_H__
#define __MDFILU_TEST_H__

#include <NSolver/MDFILU/MDFILU.h>
#include <fstream>

typedef LA::MPI::SparseMatrix SourceMatrix;
typedef SparseMatrixEZ<double> DynamicMatrix;
typedef unsigned short level_type;
typedef SparseMatrixEZ<double> LevelMatrix;
typedef unsigned short local_index_type;
typedef unsigned long global_index_type;
typedef double data_type;
typedef Vector<data_type> MDFVector;
typedef bool flag_type;

#endif
