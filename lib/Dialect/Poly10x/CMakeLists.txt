# add_mlir_dialect(Poly10x poly10x)
# set(LLVM_TARGET_DEFINITIONS Poly10x.td)
set(LLVM_TARGET_DEFINITIONS Poly10xOps.td)
mlir_tablegen(Poly10xOps.h.inc -gen-op-decls)
mlir_tablegen(Poly10xOps.cpp.inc -gen-op-defs)
mlir_tablegen(Poly10xTypes.h.inc -gen-typedef-decls -typedefs-dialect=poly10x)
mlir_tablegen(Poly10xTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=poly10x)
mlir_tablegen(Poly10xDialect.h.inc -gen-dialect-decls -dialect=poly10x)
mlir_tablegen(Poly10xDialect.cpp.inc -gen-dialect-defs -dialect=poly10x)
add_public_tablegen_target(MLIRPoly10xIncGen)
add_dependencies(mlir-headers MLIRPoly10xIncGen)

add_mlir_doc(Poly10x Poly10x Poly10x/ -gen-dialect-doc)

add_mlir_dialect_library(MLIRPoly10x
    Poly10xDialect.cpp
    Poly10xOps.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/lib/Dialect/Poly10x

    LINK_LIBS PUBLIC
) 