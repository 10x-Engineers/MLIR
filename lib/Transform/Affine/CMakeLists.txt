# add_mlir_dialect_library(AffineFullUnroll
#     AffineFullUnroll.h
#     AffineFullUnroll.cpp

#     ${PROJECT_SOURCE_DIR}/lib/Transform/Affine/
#     ADDITIONAL_HEADER_DIRS
#     LINK_LIBS PUBLIC
# )

add_mlir_library(AffineFullUnroll
    AffineFullUnroll.cpp

    ${PROJECT_SOURCE_DIR}/lib/Transform/Affine/
    ADDITIONAL_HEADER_DIRS
    LINK_LIBS PUBLIC
)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Affine)
add_public_tablegen_target(MLIRAffineFullUnrollPasses)
add_mlir_doc(Passes AffinePasses ./ -gen-pass-doc)