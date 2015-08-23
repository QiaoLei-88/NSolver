


#ifndef __NSolver__MapMatrix__
#define __NSolver__MapMatrix__

#include <NSolver/types.h>
#include <map>
#include <vector>
#include <boost/container/map.hpp>
#include <deal.II/base/utilities.h>

using namespace dealii;

template<typename Number>
class MapMatrix
{
public:
  typedef unsigned long size_type;

private:
  typedef LA::MPI::SparseMatrix SourceMatrix;
  typedef std::pair<const size_type, Number> Entry;
  typedef boost::container::map<size_type, Number> Row;
  typedef std::vector<size_type> CompressedRowPattern;
  typedef std::vector<Number> CompressedRowData;

public:
  class Accessor
  {
  public:
    Accessor (const Accessor &it);
    Accessor (const typename Row::const_iterator &it);

    const Accessor &operator++ ();
    bool operator== (const Accessor &op) const;
    bool operator!= (const Accessor &op) const;

    size_type column() const;
    const Number &value() const;

  public:
    typename Row::const_iterator map_it;
  };
  class Iterator
  {
  public:
    Iterator (const Iterator &it);
    Iterator (const typename Row::const_iterator &it);

    const Iterator &operator++ ();
    bool operator< (const Iterator &op) const;
    bool operator== (const Iterator &op) const;
    bool operator!= (const Iterator &op) const;

    const Accessor *operator-> () const;

  public:
    Accessor accessor;
  };

  typedef Iterator iterator;
  typedef Iterator const_iterator;

  MapMatrix();
  MapMatrix (const size_type row_szie,
             const size_type column_size,
             const size_type fake);
  ~MapMatrix();

  void copy_from (const SourceMatrix &matrix, bool);
  void clear();
  void reinit (const size_type row_szie,
               const size_type column_size,
               const size_type fake);
  Number el (const size_type i, const size_type j) const;
  void set (const size_type i, const size_type j, const Number value, bool fake=true);

  /*
   * Query element (@p i, @p j) in the map data structure.
   * Use this function when you know the row is not compressed to avoid if evaluation.
   */
  Number map_el (const size_type i, const size_type j) const;
  /*
   * Set @p value to element (@p i, @p j) n the map data structure.
   * Use this function when you know the row is not compressed to avoid if evaluation.
   */
  void map_set (const size_type i, const size_type j, const Number value);

  const Iterator begin (const size_type row) const;
  const Iterator end (const size_type row) const;
  const Iterator diagonal (const size_type row) const;

  void compress (const size_type row);
  const size_type *begin_compressed_pattern (const size_type row) const;
  const Number *begin_compressed_data (const size_type row) const;
  const size_type *end_compressed_pattern (const size_type row) const;
  const Number *end_compressed_data (const size_type row) const;


  size_type n_nonzero_elements() const;
  void print (std::ostream &out) const;

private:
  std::vector<Row> data;
  std::vector<CompressedRowPattern> compressed_pattern;
  std::vector<CompressedRowData> compressed_data;
};

//----------------------------------------------------------------------------//
template<typename Number>
MapMatrix<Number>:: MapMatrix()
{}

template<typename Number>
MapMatrix<Number>::MapMatrix (const size_type row_szie,
                              const size_type ,
                              const size_type)
{
  data.resize (row_szie, Row());
  compressed_pattern.resize (row_szie, CompressedRowPattern());
  compressed_data.resize (row_szie, CompressedRowData());
}

template<typename Number>
void
MapMatrix<Number>::copy_from (const SourceMatrix &M, bool)
{
  reinit (M.m(), 42, 42);

  for (size_type row = 0; row < M.m(); ++row)
    {
      const typename SourceMatrix::const_iterator end_row = M.end (row);
      for (typename SourceMatrix::const_iterator entry = M.begin (row);
           entry != end_row; ++entry)
        {
          set (row, entry->column(), entry->value());
        }
    }

  return;
}

template<typename Number>
MapMatrix<Number>::~MapMatrix()
{
  this->clear();
}
//
// Member function for MapMatrix<Number>
//
template<typename Number>
inline
void
MapMatrix<Number>::clear()
{
  for (size_type i=0; i<data.size(); ++i)
    {
      data[i].clear();
      compressed_pattern[i].clear();
      compressed_data[i].clear();
    }
  data.clear();
  compressed_pattern.clear();
  compressed_data.clear();
  return;
}

template<typename Number>
inline
void
MapMatrix<Number>::reinit (const size_type row_szie,
                           const size_type ,
                           const size_type)
{
  data.resize (row_szie, Row());
  compressed_pattern.resize (row_szie, CompressedRowPattern());
  compressed_data.resize (row_szie, CompressedRowData());
  return;
}

template<typename Number>
inline
Number
MapMatrix<Number>::el (const size_type i, const size_type j) const
{
  if (compressed_data[i].size() > 0)
    {
      // Then look into compressed data
      std::vector<size_type>::const_iterator
      it = Utilities::lower_bound (compressed_pattern[i].begin(),
                                   compressed_pattern[i].end(),
                                   j);
      if (it != compressed_pattern[i].end() && *it == j)
        {
          return (compressed_data[i][std::distance (compressed_pattern[i].begin(), it)]);
        }
      else
        {
          return (0.0);
        }
    }
  else
    {
      return (map_el (i,j));
    }
}

template<typename Number>
inline
void
MapMatrix<Number>::set (const size_type i, const size_type j, const Number value, bool)
{
  if (compressed_data[i].size() > 0)
    {
      // Then look into compressed data
      std::vector<size_type>::iterator
      it = Utilities::lower_bound (compressed_pattern[i].begin(),
                                   compressed_pattern[i].end(),
                                   j);
      Assert (it != compressed_pattern[i].end(),
              ExcMessage ("Can not insert new entry after compress!"));

      compressed_data[i][std::distance (compressed_pattern[i].begin(), it)] = value;
    }
  else
    {
      data[i][j] = value;
      return;
    }
}

template<typename Number>
inline
Number
MapMatrix<Number>::map_el (const size_type i, const size_type j) const
{
  typename Row::const_iterator ps = data[i].find (j);
  if (ps == data[i].end())
    {
      return (0.0);
    }
  else
    {
      return (ps->second);
    }
}

template<typename Number>
inline
void
MapMatrix<Number>::map_set (const size_type i, const size_type j, const Number value)
{
  data[i][j] = value;
  return;
}

template<typename Number>
inline
const typename MapMatrix<Number>::Iterator
MapMatrix<Number>::begin (const size_type row) const
{
  typename Row::const_iterator it = data[row].begin();
  Iterator rv (it);
  return (rv);
}

template<typename Number>
inline
const typename MapMatrix<Number>::Iterator
MapMatrix<Number>::end (const size_type row) const
{
  typename Row::const_iterator it = data[row].end();
  Iterator rv (it);
  return (rv);
}

template<typename Number>
inline
const typename MapMatrix<Number>::Iterator
MapMatrix<Number>::diagonal (const size_type row) const
{
  typename Row::const_iterator it = data[row].find (row);
  Iterator rv (it);
  return (rv);
}

template<typename Number>
inline
void MapMatrix<Number>::compress (const size_type row)
{
  typename Row::const_iterator it = data[row].begin();
  const typename Row::const_iterator it_end
    = data[row].end();
  const size_type size = data[row].size();
  compressed_data[row].resize (size);
  compressed_pattern[row].resize (size);

  Number *p_value = & (compressed_data[row][0]);
  size_type *p_column = & (compressed_pattern[row][0]);

  for (; it != it_end; ++it, ++p_value, ++p_column)
    {
      *p_column = it->first;
      *p_value  = it->second;
    }
  data[row].clear();
  return;
}

template<typename Number>
inline
const typename MapMatrix<Number>::size_type *
MapMatrix<Number>::begin_compressed_pattern (const size_type row) const
{
  return (& (compressed_pattern[row][0]));
}

template<typename Number>
inline
const Number *
MapMatrix<Number>::begin_compressed_data (const size_type row) const
{
  return (& (compressed_data[row][0]));
}

template<typename Number>
inline
const typename MapMatrix<Number>::size_type *
MapMatrix<Number>::end_compressed_pattern (const size_type row) const
{
  return (& (compressed_pattern[row][0])+compressed_pattern[row].size());
}

template<typename Number>
inline
const Number *
MapMatrix<Number>::end_compressed_data (const size_type row) const
{
  return (& (compressed_data[row][0])+compressed_data[row].size());
}


template<typename Number>
inline
typename MapMatrix<Number>::size_type
MapMatrix<Number>::n_nonzero_elements() const
{
  size_type n_nonzero (0);
  for (size_type i=0; i<data.size(); ++i)
    {
      n_nonzero += compressed_data[i].size();
      n_nonzero += data[i].size();
    }
  return (n_nonzero);
}

template<typename Number>
inline
void MapMatrix<Number>::print (std::ostream &out) const
{
  for (size_type i=0; i<data.size(); ++i)
    {
      if (compressed_data[i].size() != 0)
        {
          typename CompressedRowData::const_iterator it_data
            = compressed_data[i].begin();
          typename CompressedRowPattern::const_iterator it_pattern
            = compressed_pattern[i].begin();
          for (; it_data!=compressed_data[i].end(); ++it_data, ++it_pattern)
            {
              out << i << '\t'
                  << *it_pattern << '\t'
                  << *it_data << '\n';
            }
        }
      else
        {
          for (typename Row::const_iterator it = data[i].begin();
               it!=data[i].end(); ++it)
            {
              out << i << '\t'
                  << it->first << '\t'
                  << it->second << '\n';
            }
        }
    }
  return;
}

//
// Member function for MapMatrix<Number>::Accessor
//
template<typename Number>
inline
MapMatrix<Number>::Accessor:: Accessor (const Accessor &it)
  :
  map_it (it.map_it)
{}


template<typename Number>
inline
MapMatrix<Number>::Accessor::Accessor (const typename Row::const_iterator &it)
  :
  map_it (it)
{}

template<typename Number>
inline
const typename MapMatrix<Number>::Accessor &
MapMatrix<Number>::Accessor::operator++ ()
{
  ++map_it;
  return (*this);
}

template<typename Number>
inline
bool
MapMatrix<Number>::Accessor::operator== (const Accessor &op) const
{
  return (this->map_it == op.map_it);
}

template<typename Number>
inline
bool
MapMatrix<Number>::Accessor::operator!= (const Accessor &op) const
{
  return (this->map_it != op.map_it);
}

template<typename Number>
inline
typename MapMatrix<Number>::size_type
MapMatrix<Number>::Accessor:: column() const
{
  return (map_it->first);
}

template<typename Number>
inline
const Number &
MapMatrix<Number>::Accessor::value() const
{
  return (map_it->second);
}

//
// Member function for MapMatrix<Number>::Iterator
//
template<typename Number>
inline
MapMatrix<Number>::Iterator::Iterator (const Iterator &it)
  :
  accessor (it.accessor)
{}


template<typename Number>
inline
MapMatrix<Number>::Iterator::Iterator (const typename Row::const_iterator &it)
  :
  accessor (it)
{}

template<typename Number>
inline
const typename MapMatrix<Number>::Iterator &
MapMatrix<Number>::Iterator::operator++ ()
{
  ++accessor;
  return (*this);
}

template<typename Number>
inline
bool MapMatrix<Number>::Iterator::operator< (const Iterator &op) const
{
  return (this->accessor != op.accessor);
}

template<typename Number>
inline
bool MapMatrix<Number>::Iterator::operator== (const Iterator &op) const
{
  return (this->accessor == op.accessor);
}

template<typename Number>
inline
bool MapMatrix<Number>::Iterator::operator!= (const Iterator &op) const
{
  return (this->accessor != op.accessor);
}


template<typename Number>
inline
const typename MapMatrix<Number>::Accessor *
MapMatrix<Number>::Iterator::operator-> () const
{
  return (&accessor);
}

#endif
