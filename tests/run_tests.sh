cd build
export STATUS="FAILED"
make && export STATUS="PASSED"
cd ..

if [ "$STATUS" = "PASSED" ] ; then
    echo $STATUS
    cp build/hesaffexe .
    ./hesaffexe lena.png
else
    echo $STATUS
fi
