#include "lib/Transform/Arith/MulToAdd.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <cstdint>

namespace mlir {
namespace dummy {

using arith::AddIOp;
using arith::ConstantOp;
using arith::MulIOp;

// Replace y = C*x with y = C/2*x + C/2*x, when C is a power of 2, otherwise do
// nothing.
struct PowerOfTwoExpand : public OpRewritePattern<MulIOp> {
  PowerOfTwoExpand(mlir::MLIRContext *context)
      : OpRewritePattern<MulIOp>(context, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(MulIOp op,
                                PatternRewriter &rewriter) const override {
    // This pass runs on MulOp
    // get the LHS and RHS operands of the said MulOp
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);

    // canonicalization patterns ensure the constant is on the right, if there
    // is a constant See
    // https://mlir.llvm.org/docs/Canonicalization/#globally-applied-rules

    // get the defining operation of the RHS operand, should be a ConstantIntOp
    auto rhsDefiningOp = rhs.getDefiningOp<arith::ConstantIntOp>();
    if (!rhsDefiningOp)
      return failure();

    // get the value of the constant
    int64_t rhsValue = rhsDefiningOp.value();

    bool is_power_of_two = (rhsValue & (rhsValue - 1)) == 0;

    if (!is_power_of_two)
      return failure();

    // create a new constant with the value of the constant divided by 2
    ConstantOp newConstant = rewriter.create<ConstantOp>(
        rhsDefiningOp.getLoc(),
        rewriter.getIntegerAttr(rhs.getType(), rhsValue / 2));

    MulIOp newMul = rewriter.create<MulIOp>(op.getLoc(), lhs, newConstant);
    AddIOp newAdd = rewriter.create<AddIOp>(op.getLoc(), newMul, newMul);

    rewriter.replaceOp(op, newAdd);
    rewriter.eraseOp(rhsDefiningOp);
    return success();
  }
};

//
struct PeelFromMul : public OpRewritePattern<MulIOp> {
  PeelFromMul(mlir::MLIRContext *context)
      : OpRewritePattern<MulIOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(MulIOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getOperand(0);
    Value rhs = op.getOperand(1);
    auto rhsDefiningOp = rhs.getDefiningOp<arith::ConstantIntOp>();
    if (!rhsDefiningOp) {
      return failure();
    }

    int64_t value = rhsDefiningOp.value();

    // We are guaranteed `value` is not a power of two, because the greedy
    // rewrite engine ensures the PowerOfTwoExpand pattern is run first, since
    // it has higher benefit.

    ConstantOp newConstant = rewriter.create<ConstantOp>(
        rhsDefiningOp.getLoc(),
        rewriter.getIntegerAttr(rhs.getType(), value - 1));
    MulIOp newMul = rewriter.create<MulIOp>(op.getLoc(), lhs, newConstant);
    AddIOp newAdd = rewriter.create<AddIOp>(op.getLoc(), newMul, lhs);

    rewriter.replaceOp(op, newAdd);
    rewriter.eraseOp(rhsDefiningOp);

    return success();
  }
};

void MulToAddPass::runOnOperation() {
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<PowerOfTwoExpand>(&getContext());
  patterns.add<PeelFromMul>(&getContext());

  (void)applyPatternsGreedily(getOperation(), std::move(patterns));
}

} // namespace dummy
} // namespace mlir