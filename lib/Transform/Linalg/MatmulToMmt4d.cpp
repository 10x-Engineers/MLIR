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

static llvm::cl::opt<int64_t>
    clMTile("dummy-m-tile", llvm::cl::desc("Inner tile size of M dimension"),
            llvm::cl::init(32));

static llvm::cl::opt<int64_t>
    clNTile("dummy-n-tile", llvm::cl::desc("Inner tile size of N dimension"),
            llvm::cl::init(32));

static llvm::cl::opt<int64_t>
    clKTile("dummy-k-tile", llvm::cl::desc("Inner tile size of K dimension"),
            llvm::cl::init(32));

struct Matmul : public OpRewritePattern<linalg::MatmulOp> {
    Matmul(mlir::MLIRContext *context)
        : OpRewritePattern<linalg::MatmulOp>(context, /*benefit=*/1) {}

    LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                  PatternRewriter &rewriter) const override {
        int64_t M0 = clMTile;
        int64_t N0 = clNTile;
        int64_t K0 = clKTile;

        // DPS here means Destination Passing Style
        // retrieves the input operands
        auto inputs = op.getDpsInputOperands();
        // retrieves the DPS accumulator/init
        auto outputs = op.getDpsInits();

        // gets the type of given tensor by casting it to RankedTensorType
        auto lhsType = cast<RankedTensorType>(inputs[0]->get().getType());
        auto rhsType = cast<RankedTensorType>(inputs[1]->get().getType());
        auto resultType = cast<RankedTensorType>(outputs[0].getType());

        Location loc = op.getLoc();
        Value paddingValue = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getZeroAttr(lhsType.getElementType()));

        // returns the dimension of given tensor value
        llvm::SmallVector<OpFoldResult> lhsSourceDims =
            tensor::getMixedSizes(rewriter, loc, inputs[0]->get());
        // returns the ArrayAttr as a OpFoldResult
        llvm::SmallVector<OpFoldResult> lhsTileSizes =
            getAsOpFoldResult(rewriter.getI64ArrayAttr({M0, K0}));
        SmallVector<int64_t> lhsInnerDimsPos = {0, 1};
        // returns the shape that the pack result would result in
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

        // ValueRange is just a view over the underlying data
        // It does not hold the actual ownership of the data
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

        // This repalces the uses of MatmulOp with UnpackOp
        rewriter.replaceAllOpUsesWith(op, unpack);
        // erases the MatmulOp
        rewriter.eraseOp(op);

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
