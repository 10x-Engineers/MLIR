add_mlir_dialect_library(MatmulToMmt4d
    MatmulToMmt4d.h
    MatmulToMmt4d.cpp

    ${PROJECT_SOURCE_DIR}/lib/Transform/Linalg/
    ADDITIONAL_HEADER_DIRS
    LINK_LIBS PUBLIC
)