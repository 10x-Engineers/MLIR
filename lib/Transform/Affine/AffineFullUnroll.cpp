#include "lib/Transform/Affine/AffineFullUnroll.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"

namespace mlir {
namespace dummy {

using mlir::affine::AffineForOp;
using mlir::affine::loopUnrollFull;

void AffineFullUnrollPass::runOnOperation() {
  // getOperation() returns a FuncOp (this pass is registered on a FuncOp)
  // walk() will walk through all the operations in the function which are AffineForOp
  // unrolls each AffineForOp
  // if unrolling fails, emit an error and signal pass failure
  getOperation().walk([&](AffineForOp op) {
    if (failed(loopUnrollFull(op))) {
      op.emitError("unrolling failed");
      signalPassFailure();
    }
  });
}

} // namespace dummy
} // namespace mlir