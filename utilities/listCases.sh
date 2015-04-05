#!/bin/bash

# This script generates a list of all test cases for one particular test category.
# 
# Usage: execute this script inside the test category.
# For example, form the root directory of this repository, execute commands:
# $ cd tests/d2_naca2412/
# $ ../../utilities/listCases.sh
# Then the summation file test_cases.sum will be generated under the path
# 'tests/d2_naca2412/'.

WPATH=`pwd`

if [ -e test_cases.sum ]; then
  rm test_cases.sum
fi

for fold in $(ls -d */); do
  echo ${fold%%/} >> test_cases.sum
  cd ${fold}

  echo  "  builds-on:" >>  ../test_cases.sum
  while read p; do
  	echo "    "$p  >>  ../test_cases.sum
  done < builds-on

  echo  "  introduction:" >>  ../test_cases.sum
  while read p; do
  	echo "    "$p  >>  ../test_cases.sum
  done < intro.dox

  read compar < builds-on
  if [ ! -z "$compar" ]; then
	 echo  "  diff in inp:" >>  ../test_cases.sum
 	 diff  ../${compar}/input.prm input.prm >> ../test_cases.sum
  fi

  echo -e '\n'  >> ../test_cases.sum

  cd $WPATH
done
