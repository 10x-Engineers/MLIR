#!/bin/sh

mkdir -p build

pushd build

LLVM_BUILD_DIR=$1

cmake -G Ninja .. \
    -DLLVM_DIR="$LLVM_BUILD_DIR/lib/cmake/llvm" \
    -DMLIR_DIR="$LLVM_BUILD_DIR/lib/cmake/mlir" \
    -DCMAKE_BUILD_TYPE=Debug                    \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache          \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache

popd

cmake --build ./build --target check-mlir
cmake --build ./build --target dummy-opt

ln -fs ./build/compile_commands.json