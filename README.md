
## Git Clone
<pre>git clone https://github.com/ramcharanBb/DLcompiler.git</pre>


## Build Command
<pre>mkdir build && cd build<br />
cmake .. \-DMLIR_DIR=~/Desktop/llvm-project/build/lib/cmake/mlir \-DLLVM_DIR=~/Desktop/llvm-project/build/lib/cmake/llvm \-DCMAKE_BUILD_TYPE=Release<br />
make </pre>

## Test Command
<pre>./build/tools/nova-opt/nova-opt test.mlir</pre>

