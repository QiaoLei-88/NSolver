#include "../NacaBnd_test.h"

using namespace dealii;
using namespace NSFEMSolver;

int main()
{
  const BndNaca4DigitSymm NACA_foil_boundary (0012, 1.0);
  NACA_foil_boundary.test();

  return (0);
}
