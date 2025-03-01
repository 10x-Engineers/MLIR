add_mlir_dialect_library(MulToAdd
    MulToAdd.h
    MulToAdd.cpp

    ${PROJECT_SOURCE_DIR}/lib/Transform/Arith/
    ADDITIONAL_HEADER_DIRS
    LINK_LIBS PUBLIC
)