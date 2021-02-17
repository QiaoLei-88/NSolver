#!/bin/bash

# This script generates lists for each particular test category.
#
# Usage: execute this script inside the test category.
# For example, form the root directory of this repository, execute commands:
# $ ./utilities/listCases.sh

if test ! -d tests ; then
  echo "*** This script must be run from the top-level directory."
  exit
fi

cd tests

SOURCE_DIR=`pwd`
for category in $(ls -d */); do
  cd ${category}

  WPATH=`pwd`
  echo "Processing: " ${category}
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
    if [ ! -z "$compar" ] && [ -f input.prm ]; then
  	 echo  "  diff in inp:" >>  ../test_cases.sum
   	 diff ../${compar}/input.prm input.prm >> ../test_cases.sum
    fi

    echo -e '\n'  >> ../test_cases.sum

    cd $WPATH
  done

cd $SOURCE_DIR
done
