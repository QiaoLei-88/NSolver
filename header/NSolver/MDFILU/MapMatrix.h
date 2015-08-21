


#ifndef __NSolver__MapMatrix__
#define __NSolver__MapMatrix__

#include <NSolver/types.h>
#include <map>
#include <vector>

using namespace dealii;

template<typename Number>
class MapMatrix
{
public:
  typedef unsigned int size_type;

private:
  typedef LA::MPI::SparseMatrix SourceMatrix;
  typedef std::pair<const size_type, Number> Entry;
  typedef std::map<size_type, Number> Row;

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

  const Iterator begin (const size_type row) const;
  const Iterator end (const size_type row) const;
  const Iterator diagonal (const size_type row) const;

  void print (std::ostream &out) const;

private:
  std::vector<Row> data;
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
  for (size_type row = 0; row < data.size(); ++row)
    {
      data[row].clear();
    }
  data.clear();
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
    }
  data.clear();
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
  return;
}

template<typename Number>
inline
Number
MapMatrix<Number>::el (const size_type i, const size_type j) const
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
MapMatrix<Number>::set (const size_type i, const size_type j, const Number value, bool)
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
void MapMatrix<Number>::print (std::ostream &out) const
{
  for (size_type i=0; i<data.size(); ++i)
    {
      for (typename Row::const_iterator it = data[i].begin();
           it!=data[i].end(); ++it)
        {
          out << i << '\t'
              << it->first << '\t'
              << it->second << '\n';
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
