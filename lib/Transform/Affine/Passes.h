#ifndef LIB_TRANSFORM_AFFINE_PASSES_H_
#define LIB_TRANSFORM_AFFINE_PASSES_H_

#include "lib/Transform/Affine/AffineFullUnroll.h"

namespace mlir {
namespace dummy {

#define GEN_PASS_REGISTRATION
#include "lib/Transform/Affine/Passes.h.inc"

} // namespace dummy
} // namespace mlir

#endif // LIB_TRANSFORM_AFFINE_PASSES_H_