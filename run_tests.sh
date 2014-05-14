export CWD=$(pwd)

# FIXME: Weird directory dependency
export PATHSEP=$(python -c "import os; print(os.pathsep)")
export HESAFF_DIR=../hesaff
export PYTHONPATH=$PYTHONPATH$PATHSEP$CWD$PATHSEP$HESAFF_DIR$PATHSEP$HESAFF_DIR/pyhesaff
echo $PYTHONPATH



export ARGV="--quiet --noshow $@"

PRINT_DELIMETER(){
    printf "\n#\n#\n#>>>>>>>>>>> next_test\n\n"
}

echo "BEGIN: ARGV=$ARGV"
PRINT_DELIMETER

num_passed=0
num_ran=0

export FAILED_TESTS=''

RUN_TEST()
{
    echo "RUN_TEST: $@"
    export TEST="python $@ $ARGV"
    $TEST
    export RETURN_CODE=$?
    PRINT_DELIMETER
    num_passed=$(($num_passed + (1 - $RETURN_CODE)))
    num_ran=$(($num_ran + 1))

    if [ "$RETURN_CODE" != "0" ] ; then
        export FAILED_TESTS="$FAILED_TESTS\n$TEST"
    fi

}

RUN_TEST pyhesaff/tests/test_pyhesaff_simple_iterative.py

RUN_TEST pyhesaff/tests/test_pyhesaff_simple_parallel.py



#---------------------------------------------
echo "RUN_TESTS: DONE"

if [ "$FAILED_TESTS" != "" ] ; then
    echo "-----"
    printf "Failed Tests:" 
    printf "$FAILED_TESTS\n"
    echo "-----"
fi

echo "$num_passed / $num_ran tests passed"
