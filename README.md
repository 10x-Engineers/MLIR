# MLIR

## Build MLIR
```bash
git clone https://github.com/llvm/llvm-project.git

cd llvm-project
mkdir build && cd build

cmake -G Ninja ../llvm              \ 
    -DLLVM_ENABLE_PROJECTS=mlir     \
    -DLLVM_BUILD_EXAMPLES=ON        \ 
    -DLLVM_ENABLE_ASSERTIONS=ON     \
    -DCMAKE_BUILD_TYPE=Release      \
    -DLLVM_ENABLE_RTTI=ON           \
    -DLLVM_TARGETS_TO_BUILD="host"  \ 
    -DCMAKE_C_COMPILER=clang        \
    -DCMAKE_CXX_COMPILER=clang++    \
    -DLLVM_ENABLE_LLD=ON            \
    -DLLVM_CCACHE_BUILD=ON          \
```
> **Note:** Further Instructions to build MLIR can be found [here](https://mlir.llvm.org/getting_started/)
