add_mlir_dialect(Poly10xOps poly10x)
add_mlir_doc(Poly10xDialect Poly10xDialect Poly10x/ -gen-dialect-doc)

add_mlir_dialect_library(MLIRPoly10x
    Poly10xDialect.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/lib/Dialect/Poly10x

    LINK_LIBS PUBLIC
) 