# This defines the CMake version to use
cmake_minimum_required(VERSION 3.20.0)

# This defines the project name and the languages used
project(mlir-tutorial LANGUAGES CXX C)

# This sets the C++ standard to use
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

# This finds the MLIRConfig.cmake file. The path to search on is provided through command-line using -DMLIR_DIR
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# This finds the AddLLVM.cmake file in the llvm-project/build/lib/cmake/llvm directory
include(AddLLVM)

# Add subdirectories to the build folder
add_subdirectory(tests)