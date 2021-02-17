#include <NSolver/print_version.h>

#include <version.info>

void
print_version(std::ostream &os)
{
  std::string build_type(BUILD_TYPE);
  std::string fill(22 - build_type.size(), ' ');

  os << "\n\n"
     << "x----------------------------"
     << "---------x\n"
     << "|             FEMNSolver     "
     << "         |\n"
     << "|       " << VERSION_AND_HASH << "        |\n"
     << "|   Built time: " << BUILD_TIME << "   |\n"
     << "|   Build type: " << build_type << fill << "|\n"
     << "x----------------------------"
     << "---------x\n\n"
     << std::endl;

  return;
}
