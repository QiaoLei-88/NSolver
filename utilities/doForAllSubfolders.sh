#
# A script for doing something in all sub-folders.
# May be usefull in updating testcases.
#

WPATH=`pwd`

for fold in $(ls -d */); do
  pushd .
  cd $fold
#  patch input.prm < ../input.patch
  popd
done

