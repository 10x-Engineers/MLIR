# This defines the CMake version to use
cmake_minimum_required(VERSION 3.20.0)

# This defines the project name and the languages used
project(mlir-tutorial LANGUAGES CXX C)

# This is turned on to export the compiled commands in compile_commands.json file so that clangd can use it
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# This sets the C++ standard to use
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

# This finds the MLIRConfig.cmake file. The path to search on is provided through command-line using -DMLIR_DIR
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# This finds the AddLLVM.cmake file in the llvm-project/build/lib/cmake/llvm directory
include(AddLLVM)

# The $MLIR_CMAKE_DIR variable is set in the MLIRConfig.cmake file
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

# This finds the AddMLIR.cmake file in the llvm-project/build/lib/cmake/mlir directory
include(AddMLIR)

# The $LLVM_INCLUDE_DIRS and $MLIR_INCLUDE_DIRS variable is set in the LLVMConfig.cmake and MLIRConfig.cmake file respectively
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

include_directories("./")

# Add subdirectories to the build folder
add_subdirectory(tests)
add_subdirectory(tools)