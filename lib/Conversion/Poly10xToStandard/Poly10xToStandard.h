#ifndef LIB_CONVERSION_POLY10XTOSTANDARD_POLY10XTOSTANDARD_H_
#define LIB_CONVERSION_POLY10XTOSTANDARD_POLY10XTOSTANDARD_H_

#include "mlir/Pass/Pass.h"

// Extra includes needed for dependent dialects
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace mlir {
namespace dummy {
namespace poly10x {

#define GEN_PASS_DECL
#include "lib/Conversion/Poly10xToStandard/Poly10xToStandard.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Conversion/Poly10xToStandard/Poly10xToStandard.h.inc"

} // namespace poly10x
} // namespace dummy
} // namespace mlir

#endif // LIB_CONVERSION_POLY10XTOSTANDARD_POLY10XTOSTANDARD_H_
