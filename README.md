# MLIR
This tutorial is supposed to be done while reading the articles mentioned here: https://github.com/j2kun/mlir-tutorial?tab=readme-ov-file#articles. Each PR in this repo implements some part of those articles. One can look at the original repository and even follow that as most of this is implemented from there. Purpose of this tutorial is to get one familiarized with how to set up a project that uses MLIR and what are general MLIR concepts like Dialects, Operations, and most important of all progressive lowering.
## Build MLIR
```bash
git clone https://github.com/llvm/llvm-project.git
git checkout 9f688117271e95808342e9609c5cb377604f1278

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

## Build Tutorial
```bash
git clone https://github.com/10x-Engineers/MLIR.git

bash _build.sh <path/to/llvm-project/build>
```

## Build `dummy-opt`
```bash
cmake --build ./build --target dummy-opt
```
> **Note:** The script `_build.sh` already builds `dummy-opt`

## Test
```bash
cmake --build ./build --target check-mlir
```

## References
- [MLIR Official Documentation](https://mlir.llvm.org/)
- [j2kun/mlir-tutorial](https://github.com/j2kun/mlir-tutorial)
- [vguerra/mlir-tutorial-cmake](https://github.com/vguerra/mlir-tutorial-cmake)
