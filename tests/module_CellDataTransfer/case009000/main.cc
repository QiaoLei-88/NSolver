#include "../CellDataTransfer_test.h"

template <int dim>
void test();

std::ofstream fout;

int main (int, char **)
{
  fout.open ("output.out");

  test<2>();
  fout << std::endl << std::endl;
  test<3>();

  fout.close();
  return (0);
}


template <int dim>
void test()
{
  typedef Triangulation<dim> TriaType;
  typedef unsigned int size_type;

  TriaType tria;
  GridGenerator::hyper_cube (tria);

  size_type n_active = tria.n_active_cells();
  std::vector<double> v_double (n_active);
  std::vector<float> v_float (n_active);

  for (size_type i=0; i<n_active; ++i)
    {
      v_double[i] = 83.41;
      v_float[i] = 42;
    }

  CellDataTransfer<dim> cell_data_transfer (tria);

  cell_data_transfer.push_back (&v_double[0],
                                v_double.size());
  cell_data_transfer.push_back (&v_float[0],
                                v_float.size());
  tria.refine_global (2);

  n_active = tria.n_active_cells();
  v_double.resize (n_active);
  v_float.resize (n_active);

  cell_data_transfer.get_transfered_data (0, &v_double[0]);
  cell_data_transfer.get_transfered_data (1, &v_float[0]);

  cell_data_transfer.clear();

  for (size_type i=0; i<n_active; ++i)
    {
      fout << v_double[i] << ", " << v_float[i]
           << std::endl;
    }

  return;
}
