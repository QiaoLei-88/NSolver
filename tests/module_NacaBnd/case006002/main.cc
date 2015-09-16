#include "../NacaBnd_test.h"

using namespace dealii;
using namespace NSFEMSolver;

#define dim 2

int main()
{
  Triangulation<dim> triangulation;
  {
    GridIn<dim> grid_in;
    grid_in.attach_triangulation (triangulation);

    std::ifstream input_file ("NACA0012.msh");
    Assert (input_file, ExcFileNotOpen ("NACA0012.msh"));
    grid_in.read_msh (input_file);
  }

  const BndNaca4DigitSymm NACA_foil_boundary (0012, 1.0);
  std::ofstream fout ("output.out");
  Assert (fout, ExcFileNotOpen ("output.out"));
  fout << std::scientific;
  fout.precision (8);

  for (typename Triangulation<dim>::active_cell_iterator
       cell = triangulation.begin_active();
       cell != triangulation.end();
       ++cell)
    for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
      if (cell->face (f)->at_boundary())
        {
          if (cell->face (f)->boundary_id() == 1)
            {
              typename Boundary<2,2>::FaceVertexNormals face_vertex_normals;
              NACA_foil_boundary.get_normals_at_vertices (cell->face (f),
                                                          face_vertex_normals);
              for (unsigned int v=0; v<GeometryInfo<2>::vertices_per_face; ++v)
                {
                  fout << cell->face (f)->vertex (v) << '\t';
                  fout << face_vertex_normals[v] << std::endl;
                }
            }
        }

  fout.close();
  return (0);
}
