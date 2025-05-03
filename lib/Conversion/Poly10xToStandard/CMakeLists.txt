set(LLVM_TARGET_DEFINITIONS Poly10xToStandard.td)
mlir_tablegen(Poly10xToStandard.h.inc -gen-pass-decls)

add_public_tablegen_target(MLIRPoly10xToStandardPassIncGen)
add_dependencies(mlir-headers MLIRPoly10xToStandardPassIncGen)

add_mlir_doc(Poly10xToStandard Poly10xToStandard Poly10xToStandard/ -gen-pass-doc)

add_mlir_library(MLIRPoly10xToStandard
    Poly10xToStandard.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/lib/Conversion/Poly10xToStandard

    DEPENDS
    MLIRPoly10xToStandardPassIncGen

    LINK_LIBS PUBLIC
)