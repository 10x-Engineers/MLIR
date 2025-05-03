# dialect, ops, and types go in same tablegen_target
set(LLVM_TARGET_DEFINITIONS Poly10xOps.td)
mlir_tablegen(Poly10xOps.h.inc -gen-op-decls)
mlir_tablegen(Poly10xOps.cpp.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS Poly10xTypes.td)
mlir_tablegen(Poly10xTypes.h.inc -gen-typedef-decls -typedefs-dialect=poly10x)
mlir_tablegen(Poly10xTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=poly10x)

set(LLVM_TARGET_DEFINITIONS Poly10x.td)
mlir_tablegen(Poly10xDialect.h.inc -gen-dialect-decls -dialect=poly10x)
mlir_tablegen(Poly10xDialect.cpp.inc -gen-dialect-defs -dialect=poly10x)

add_public_tablegen_target(MLIRPoly10xIncGen)

# specifiying that mlir-headers depends on the above generated files
add_dependencies(mlir-headers MLIRPoly10xIncGen)

# canonicalization patterns go in a separate tablegen_target
set(LLVM_TARGET_DEFINITIONS Poly10xPatterns.td)
mlir_tablegen(Poly10xCanonicalize.cpp.inc -gen-rewriters)
add_public_tablegen_target(MLIRPoly10xCanonicalizationIncGen)

# docs
add_mlir_doc(Poly10x Poly10x Poly10x/ -gen-dialect-doc)

add_mlir_dialect_library(MLIRPoly10x
    Poly10xDialect.cpp
    Poly10xOps.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/lib/Dialect/Poly10x

    DEPENDS
    MLIRPoly10xIncGen

    LINK_LIBS PUBLIC
)