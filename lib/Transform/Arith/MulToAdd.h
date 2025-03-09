#ifndef LIB_TRANSFORM_ARITH_MULTOADD_H_
#define LIB_TRANSFORM_ARITH_MULTOADD_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace dummy {

class MulToAddPass
    : public PassWrapper<MulToAddPass, OperationPass<mlir::func::FuncOp>> {
  private:
    void runOnOperation() override;

    StringRef getArgument() const final { return "mul-to-add"; }

    StringRef getDescription() const final {
        return "Convert multiplications to repeated additions";
    }
};

} // namespace dummy
} // namespace mlir

#endif // LIB_TRANSFORM_ARITH_MULTOADD_H_