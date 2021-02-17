#!/bin/bash

# Expected parameters
# $1 exec_diff
# $2 para_diff
# $3 reference output
# $4 test output

EXEC_DIFF=$1
PARA_DIFF=$2
REF_OUTPUT=$3
TST_OUTPUT=$4

echo -e "\nComparing output...\n"
rtcode="$(cat exec.exit)"

if [ -z "`grep "Expected to fail!" ${REF_OUTPUT}`" ];then
    # This case is expected to run normally
    eval "${EXEC_DIFF} -a 1e-6 -r 1e-8 -s ' \t\n:<>=,()[];' ${PARA_DIFF} ${REF_OUTPUT} ${TST_OUTPUT}" 2>&1 | tee output.diff
    exit ${PIPESTATUS[0]}
else
    # This case is expected to fail
    if [ "$rtcode" == "0" ]; then
        echo "DIFF: Unexpected normal exit of program."
        echo "DIFF: In this case this script should not be called."
        echo ""
        exit 1
    else
        echo "DIFF: Program failed as expected."
        echo ""
        rm -rf *.diff
        touch output.diff
        exit 0
    fi
fi

exit 0
