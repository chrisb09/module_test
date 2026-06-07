
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DWITH_AIX=ON -DWITH_PHYDLL=ON

DEBUG=1 cmake --build build -j 4