#!/bin/bash

echo "***** diff A.out *****"
diff A.out A.out.reference
echo "***** diff apply.out *****"
diff apply.out apply.out.reference
echo "***** diff LU.out *****"
diff LU.out LU.out.reference


./removeParens.sh matrix.out
./deal2MTX LU.out LU.MTX
./deal2MTX matrix.out matrix.MTX
./deal2MTX A.out A.MTX
