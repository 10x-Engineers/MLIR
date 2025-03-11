#include "lib/Transform/Linalg/MatmulToMmt4d.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>

namespace mlir {
namespace dummy {

struct Matmul : public OpRewritePattern<linalg::MatmulOp> {
    Matmul(mlir::MLIRContext *context)
        : OpRewritePattern<linalg::MatmulOp>(context, /*benefit=*/1) {}

    LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                  PatternRewriter &rewriter) const override {
        // llvm::outs() << "Matched a MatmulOp\n";
        int64_t M0 = 512;
        int64_t N0 = 32;
        int64_t K0 = 32;

        auto inputs = op.getDpsInputOperands();
        auto outputs = op.getDpsInits();

        auto lhsType = cast<RankedTensorType>(inputs[0]->get().getType());
        auto rhsType = cast<RankedTensorType>(inputs[1]->get().getType());
        auto resultType = cast<RankedTensorType>(outputs[0].getType());

        // auto lhsShape = lhsType.getShape();
        // auto rhsShape = rhsType.getShape();
        // auto resultShape = resultType.getShape();

        // llvm::outs() << "LHS Shape: " << lhsShape[0] << "x" << lhsShape[1] <<
        // '\n'; llvm::outs() << "RHS Shape: " << rhsShape[0] << "x" <<
        // rhsShape[1] << '\n'; llvm::outs() << "Result Shape: " <<
        // resultShape[0] << "x" << resultShape[1] << '\n';

        // for (OpFoldResult result : final) {
        //     // Process each OpFoldResult
        //     if (auto val = result.dyn_cast<Value>()) {
        //         llvm::outs() << "Value: " << val << "\n";
        //     } else if (auto attr = result.dyn_cast<Attribute>()) {
        //         llvm::outs() << "Attribute: " << attr << "\n";
        //     }
        // }

        Location loc = op.getLoc();
        Value paddingValue = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getZeroAttr(lhsType.getElementType()));

        llvm::SmallVector<OpFoldResult> lhsSourceDims =
            tensor::getMixedSizes(rewriter, loc, inputs[0]->get());
        llvm::SmallVector<OpFoldResult> lhsTileSizes =
            getAsOpFoldResult(rewriter.getI64ArrayAttr({M0, K0}));
        SmallVector<int64_t> lhsInnerDimsPos = {0, 1};
        SmallVector<OpFoldResult> lhsResultDims =
            linalg::PackOp::getResultShape(rewriter, loc, lhsSourceDims,
                                           lhsTileSizes, lhsInnerDimsPos,
                                           lhsInnerDimsPos);
        tensor::EmptyOp emptyOp0 = rewriter.create<tensor::EmptyOp>(
            loc, lhsResultDims, lhsType.getElementType());
        linalg::PackOp lhsPack = rewriter.create<linalg::PackOp>(
            loc, inputs[0]->get(), emptyOp0, lhsInnerDimsPos, lhsTileSizes,
            paddingValue, lhsInnerDimsPos);

        llvm::SmallVector<OpFoldResult> rhsSourceDims =
            tensor::getMixedSizes(rewriter, loc, inputs[1]->get());
        llvm::SmallVector<OpFoldResult> rhsTileSizes =
            getAsOpFoldResult(rewriter.getI64ArrayAttr({N0, K0}));
        SmallVector<int64_t> rhsInnerDimsPos = {1, 0};
        SmallVector<OpFoldResult> rhsResultDims =
            linalg::PackOp::getResultShape(rewriter, loc, rhsSourceDims,
                                           rhsTileSizes, rhsInnerDimsPos,
                                           rhsInnerDimsPos);
        tensor::EmptyOp emptyOp1 = rewriter.create<tensor::EmptyOp>(
            loc, rhsResultDims, rhsType.getElementType());
        linalg::PackOp rhsPack = rewriter.create<linalg::PackOp>(
            loc, inputs[1]->get(), emptyOp1, rhsInnerDimsPos, rhsTileSizes,
            paddingValue, rhsInnerDimsPos);

        llvm::SmallVector<OpFoldResult> resSourceDims =
            tensor::getMixedSizes(rewriter, loc, outputs[0]);
        llvm::SmallVector<OpFoldResult> resTileSizes =
            getAsOpFoldResult(rewriter.getI64ArrayAttr({M0, N0}));
        SmallVector<int64_t> resInnerDimsPos = {0, 1};
        SmallVector<OpFoldResult> resResultDims =
            linalg::PackOp::getResultShape(rewriter, loc, resSourceDims,
                                           resTileSizes, resInnerDimsPos,
                                           resInnerDimsPos);
        tensor::EmptyOp emptyOp2 = rewriter.create<tensor::EmptyOp>(
            loc, resResultDims, resultType.getElementType());
        linalg::PackOp resPack = rewriter.create<linalg::PackOp>(
            loc, outputs[0], emptyOp2, resInnerDimsPos, resTileSizes,
            paddingValue, resInnerDimsPos);

        linalg::Mmt4DOp mmt4d = rewriter.create<linalg::Mmt4DOp>(
            loc, resPack.getResult().getType(),
            ValueRange{lhsPack->getResult(0), rhsPack->getResult(0)},
            ValueRange{resPack->getResult(0)});

        llvm::SmallVector<OpFoldResult> mmt4dDims =
            tensor::getMixedSizes(rewriter, loc, mmt4d.getDpsInits()[0]);
        tensor::EmptyOp emptyOp3 = rewriter.create<tensor::EmptyOp>(
            loc, resSourceDims, resultType.getElementType());
        linalg::UnPackOp unpack = rewriter.create<linalg::UnPackOp>(
            loc, mmt4d->getResult(0), emptyOp3, resInnerDimsPos, resTileSizes,
            resInnerDimsPos);

        rewriter.replaceAllOpUsesWith(op, unpack);
        rewriter.eraseOp(op);

        // for (OpFoldResult result : mmt4dDims) {
        //     // Process each OpFoldResult
        //     if (auto val = result.dyn_cast<Value>()) {
        //         llvm::outs() << "Value: " << val << "\n";
        //     } else if (auto attr = result.dyn_cast<Attribute>()) {
        //         llvm::outs() << "Attribute: " << attr << "\n";
        //     }
        // }
        // llvm::outs() << "Hell0\n";
        return success();
    }
};

void MatmulToMmt4dPass::runOnOperation() {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.insert<Matmul>(&getContext());
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
}
} // namespace dummy
} // namespace mlir