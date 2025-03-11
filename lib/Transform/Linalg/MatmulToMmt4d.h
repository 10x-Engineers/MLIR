#ifndef LIB_TRANSFORM_ARITH_MATMULTOMMT4D_H_
#define LIB_TRANSFORM_ARITH_MATMULTOMMT4D_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace dummy {

class MatmulToMmt4dPass
    : public PassWrapper<MatmulToMmt4dPass, OperationPass<mlir::func::FuncOp>> {
  private:
    void runOnOperation() override;

    StringRef getArgument() const final { return "matmul-to-mmt4d"; }

    StringRef getDescription() const final {
        return "Convert linalg.matmul to linalg.mmt4d";
    }
};

} // namespace dummy
} // namespace mlir

#endif // LIB_TRANSFORM_ARITH_MATMULTOMMT4D_H_