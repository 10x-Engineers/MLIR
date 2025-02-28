#!/bin/sh

rm -rf build
mkdir build

pushd build

LLVM_BUILD_DIR=$1

cmake -G Ninja .. \
    -DLLVM_DIR="$LLVM_BUILD_DIR/lib/cmake/llvm" \
    -DMLIR_DIR="$LLVM_BUILD_DIR/lib/cmake/mlir" \
    -DCMAKE_BUILD_TYPE=Debug

popd