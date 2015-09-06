#!/bin/bash
#
#  This script run regression test for the untested latest local and
#  remote revision inside directory ${WORK_DIR}.
#
#  PATH needs to be setup manually for running by crontab.
#

WORK_DIR=/w/qiaolei/devel/NSolver/regTest
PATH=/u/qiaolei/Library/mpi/bin:/u/qiaolei/Library/cmake-3.0.2/bin:/u/qiaolei/.bin:/usr/NX/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/u/qiaolei/Library/cgnslib/bin

function bdie () {
  echo "**** Error in line $1"
  echo "**** Error msg: ${*:2}"
  exit 1
}

function run_tests() {
	echo -e '\n'
	echo "${HOSTNAME} : ${LAST_LOCAL_VER}" >> /u/qiaolei/.regressionTestList
	cmake ${SRC_DIR} || bdie "$LINENO" "cmake failed."
	make -j16 || bdie "$LINENO" "make failed."
	(
		START_TIME=`date`
		touch ${TEST_RUNNING} 
		ctest -j16 >> ctest.output
		rm ${TEST_RUNNING}
		echo ${START_TIME} > ${TEST_FINISHED}
		echo `date` >> ${TEST_FINISHED}
		) & # let the test run in background.
}

PRJNAME=NSolver
TEST_FINISHED=testFinished
TEST_RUNNING=testRunning
SRC_DIR=${WORK_DIR}/${PRJNAME}
TEST_DIR=${WORK_DIR}/regressionTests


# main()
source /u/qiaolei/.bashrc

N_CPU_ALL=`qusage | grep $HOSTNAME | awk '{ print $3 }'`
N_CPU_BUSSY=`qusage | grep $HOSTNAME | awk '{ print $2 }'`
N_CPU_IDLE=`echo "$N_CPU_ALL - $N_CPU_BUSSY" | bc`
N_CPU_IDLE=`printf "%.0f" $N_CPU_IDLE`

if [ ${N_CPU_IDLE} -lt 16 ]; then
  bdie "$LINENO" "Not enough idle CPUs!"
fi


cd ${WORK_DIR} || bdie "$LINENO" "${WORK_DIR} doesn't accessible!"

if [ ! -d ${PRJNAME} ]; then
	echo "  No repository detected. Clone it ..."
	git clone https://github.com/QiaoLei-88/${PRJNAME}.git
fi


if [ ! -d ${TEST_DIR} ]; then
	echo "  No test directory detected. Creating ..."
	mkdir ${TEST_DIR}
fi

# Get the latest revision of local repository
pushd . >> /dev/null
cd ${SRC_DIR}
git checkout master -q
LAST_LOCAL_VER=`git log -1 --format=format:%H`
popd >> /dev/null

# Check whether the latest local revision tested.
pushd . >> /dev/null

cd ${TEST_DIR}
if [ ! -d ${LAST_LOCAL_VER} ]; then
	LAST_LOCAL_VER_TESTED=false
else 
	cd ${LAST_LOCAL_VER}
	if [ -e ${TEST_FINISHED} ]; then
		LAST_LOCAL_VER_TESTED=true
	else 
		if [ -e ${TEST_RUNNING} ]; then
			bdie "$LINENO" "It seems there is an unfinished test."
		fi
		LAST_LOCAL_VER_TESTED=false
	fi
fi
popd >> /dev/null

# If the latest local revision is still not tested then test it.
# This usually should not happen.
pushd . >> /dev/null
if [ ${LAST_LOCAL_VER_TESTED} == 'false' ]; then
	cd ${TEST_DIR}
	if [ ! -d ${LAST_LOCAL_VER} ]; then
		mkdir ${LAST_LOCAL_VER}
		cd ${LAST_LOCAL_VER} || bdie "$LINENO" "Can not get into directory ${LAST_LOCAL_VER}."
	else
		cd ${LAST_LOCAL_VER} || bdie "$LINENO" "Can not get into directory ${LAST_LOCAL_VER}."
		rm -rf ./*
	fi

	pushd . >> /dev/null
	cd ${SRC_DIR}
	git checkout master -q
	popd >> /dev/null

	run_tests
fi
popd >> /dev/null

# Now we assume the latest local revision is tested.
# Check whether the latest version is also the latest remote revision
# If it is not identical to the latest remote revision then update and test, else
# just say something.

pushd . >> /dev/null

cd ${SRC_DIR}
git checkout master -q
NEED_UPDATE=`git ls-remote origin master | grep "${LAST_LOCAL_VER}"`

popd >> /dev/null

if [ -z "${NEED_UPDATE}" ] ; then
	pushd . >> /dev/null
	cd ${SRC_DIR}
	git checkout master -q
	git pull -q || bdie "$LINENO" "Pull remote repository failed."
	LAST_LOCAL_VER=`git log -1 --format=format:%H`
	popd >> /dev/null

	cd ${TEST_DIR}
	if [ -d ${LAST_LOCAL_VER} ]; then
		bdie "$LINENO" "How could revision ${LAST_LOCAL_VER} tested before updated??"
	fi
	mkdir ${LAST_LOCAL_VER}
	cd ${LAST_LOCAL_VER} || bdie "$LINENO" "Can not get into directory ${LAST_LOCAL_VER}."

	run_tests
fi
