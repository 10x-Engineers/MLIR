add_mlir_dialect_library(AffineFullUnroll
    AffineFullUnroll.h
    AffineFullUnroll.cpp

    ${PROJECT_SOURCE_DIR}/lib/Transform/Affine/
    ADDITIONAL_HEADER_DIRS
    LINK_LIBS PUBLIC
)