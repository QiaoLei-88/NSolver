#!/bin/bash
#
# A script to update reference outputs for on test Category.
# Useful when program has demanded changes.
#

if [ ! -f utilities/script/updateFailedReferResults.sh ]; then
  echo "*** This script must be run from the top-level directory of NSolver."
  exit 1
fi

if [ -z "$1" ]; then
    echo " Please specify the root path to regTest!"
    exit 2
fi
TESTPATH="$1"


cd tests
nUpDate=0

for testCategory in $(ls -d */); do
  pushd . > /dev/null
  cd -- "$testCategory"
  echo ${testCategory%/}

  WPATH=`pwd`
  Category=${PWD##*/}

  for subcaseName in $(ls -d */); do
    pushd . > /dev/null

    DiffOutFileName=${TESTPATH}/tests/${Category}/${subcaseName}output.diff
    if [ ! -f "${DiffOutFileName}" ] \
       || \
       [ -n "$(grep 'differs' "${DiffOutFileName}")" ] ; then
      cd -- "$subcaseName"
      printf "  ${subcaseName%/} : "

      ReferFileName=$(ls *.reference)
      OutPutFileName=${ReferFileName%\.reference}
      FullPathNewFile=${TESTPATH}tests/${Category}/${subcaseName}${OutPutFileName}
      if [ -f ${FullPathNewFile} ]; then
        if [ "x${DRYRUN}" == "x" ]; then
          cp -v ${FullPathNewFile} ${ReferFileName}
        else
          echo "DRYRUN: ${ReferFileName} <- ${FullPathNewFile}"
        fi
        nUpDate=$((nUpDate+1))
      else
        echo -e "\e[1;33mWARNING: No new output file found.\e[0m"
      fi
    else
      echo -e "  ${subcaseName%/} : \e[1;32mPASSED\e[0m, no update"
    fi
    popd > /dev/null
  done
  popd > /dev/null
done

if [ ${nUpDate} -le 1 ]; then
  echo -n -e "\n${nUpDate} reference updated"
else
  echo -n -e "\n${nUpDate} references updated"
fi
if [ "x${DRYRUN}" == "x" ]; then
  echo
else
  echo " (DRYRUN)"
fi
