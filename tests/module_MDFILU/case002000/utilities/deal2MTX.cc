#include <fstream>
#include <iostream>

using namespace std;

int main (int argc, char *argv[])
{

  if (argc<3)
    {
      cerr << "Please tell me the input and output file names in command line parameters."
           << endl;
      return (1);
    }

  unsigned n_row (0);
  unsigned n_column (0);
  int n_none_zero (-1);

  {
    std::ifstream fin (argv[1]);
    if (!fin.good())
      {
        cerr << "Cannot open the specified input file."
             << endl;
        return (2);
      }
    while (!fin.eof())
      {
        ++n_none_zero;
        unsigned i,j;
        double ignore;
        fin >> i;
        fin >> j;
        fin >> ignore;

        n_row = std::max (i, n_row);
        n_column = std::max (j, n_column);
      }
    fin.close();
  }


  std::ofstream fout (argv[2]);
  if (!fout.good())
    {
      cerr << "Cannot open the specified output file."
           << endl;
      return (3);
    }

  fout <<"%%MatrixMarket matrix coordinate real general\n";
  fout << n_row+1 << "\t" << n_column +1 << "\t"
       << n_none_zero << std::endl;

  std::ifstream fin (argv[1]);
  if (!fin.good())
    {
      cerr << "Cannot open the specified input file for second time reading."
           << endl;
      return (4);
    }

  for (int line=0; line < n_none_zero; ++line)
    {
      unsigned i,j;
      double value;
      fin >> i;
      fin >> j;
      // i,j transpose
      fout << i + 1 << "\t";
      fout << j + 1 << "\t";
      fin >> value;
      fout << value << std::endl;
    }

  fin.close();
  fout.close();

  return (0);
}
