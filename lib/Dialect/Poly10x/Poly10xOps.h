#ifndef LIB_DIALECT_POLY_POLY10XOPS_H_
#define LIB_DIALECT_POLY_POLY10XOPS_H_

#include "lib/Dialect/Poly10x/Poly10xDialect.h"
#include "lib/Dialect/Poly10x/Poly10xTypes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"

#define GET_OP_CLASSES
#include "lib/Dialect/Poly10x/Poly10xOps.h.inc"

#endif // LIB_DIALECT_POLY_POLY10XOPS_H_