# The MLIR_DIALECT_LIBS property was set in MLIRConfig.cmake file
# This retrieves the value of the property and stores it in dialect_libs variable
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

set (LIBS
    ${dialect_libs}
    AffineFullUnroll
    MulToAdd
    MLIROptLib
    MLIRPass
)

# Macro defined in AddLLVM.cmake file
add_llvm_executable(dummy-opt dummy-opt.cpp)

# Macro defined in AddLLVM.cmake file
llvm_update_compile_flags(dummy-opt)
target_link_libraries(dummy-opt PRIVATE ${LIBS})

# Macro defined in AddMLIR.cmake file
mlir_check_all_link_libraries(dummy-opt)