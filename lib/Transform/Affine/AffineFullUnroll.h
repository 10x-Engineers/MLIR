#ifndef  LIB_TRANSFORM_AFFINE_AFFINEFULLUNROLL_H_
#define  LIB_TRANSFORM_AFFINE_AFFINEFULLUNROLL_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace dummy {

class AffineFullUnrollPass
    : public PassWrapper<AffineFullUnrollPass,
                         OperationPass<mlir::func::FuncOp>> {
private:
    void runOnOperation() override;

    StringRef getArgument() const final { return "affine-full-unroll"; }

    StringRef getDescription() const final {
        return "Fully unroll all affine loops";
    }
};

} // namespace dummy
} // namespace mlir

#endif // LIB_TRANSFORM_AFFINE_AFFINEFULLUNROLL_H_