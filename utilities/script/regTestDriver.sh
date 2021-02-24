#!/bin/bash

# Expected parameters
# $1 exec_run
# $2 para_run
# $3 reference output
# $4 test output

eval "$1 $2" 2>&1 | tee screen.log
rtcode="${PIPESTATUS[0]}"
echo $rtcode > exec.exit
echo -e "\n\n"

if [ -z "`grep "Expected to abort!" $3`" ];then
    # This case is expected to run normally
    if [ "$rtcode" != "0" ]; then
      # CMake will clear target output if command exit with non-zero state
      cp $4 "$4".save
    fi
    exit $rtcode
else
    # This case is expected to fail
    if [ "$rtcode" == "0" ]; then
        # Success is unexpected.
        echo "EXEC: Program finished normally but failure is expected!"
        echo ""
        cp $4 "$4".surprise
        exit 1
    else
        echo "EXEC: Program aborted as expected."
        echo ""
        echo "Expected to abort!" >> $4
        # Then continue to compare output
        exit 0
    fi
fi
