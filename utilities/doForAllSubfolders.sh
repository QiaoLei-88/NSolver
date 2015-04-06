#
# A script for doing something in all sub-folders.
# May be usefull in updating testcases.
#

WPATH=`pwd`

for fold in $(ls -d */); do
  #git rm ${fold}/screen.log.reference
  #git add ${fold}/iter_history.out.reference
done

