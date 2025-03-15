#include "lib/Dialect/Poly10x/Poly10xOps.h"

namespace mlir {
namespace dummy {
namespace poly10x {

OpFoldResult ConstantOp::fold(ConstantOp::FoldAdaptor adaptor) {
  return adaptor.getCoefficients();
}

}  // namespace poly
}  // namespace tutorial
}  // namespace mlir