#ifndef LIB_TYPES_POLY_POLY10XTYPES_H_
#define LIB_TYPES_POLY_POLY10XTYPES_H_

// Required because the .h.inc file refers to MLIR classes and does not itself
// have any includes.
#include "mlir/IR/DialectImplementation.h"

#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/Poly10x/Poly10xTypes.h.inc"

#endif // LIB_TYPES_POLY_POLY10XTYPES_H_
