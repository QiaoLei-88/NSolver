#!/bin/bash
#
# A script for doing something in all sub-folders.
# May be usefull in updating testcases.
#

WPATH=`pwd`
Category=`echo "${PWD##*/}"`

for fold in $(ls -d */); do
  pushd . > /dev/null
  cd $fold
    echo $fold
#    patch input.prm < ../input.patch
  popd > /dev/null
done
