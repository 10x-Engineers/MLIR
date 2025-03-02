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

cmake --build ./build --target MLIRAffineFullUnrollPasses
cmake --build ./build --target mlir-doc
cmake --build ./build --target dummy-opt

cmake --build ./build --target check-mlir


ln -fs ./build/compile_commands.json
ln -fs ./build/tablegen_compile_commands.yml