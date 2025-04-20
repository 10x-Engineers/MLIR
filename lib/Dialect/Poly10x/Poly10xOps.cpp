#include "lib/Dialect/Poly10x/Poly10xOps.h"
#include "lib/Dialect/Poly10x/Poly10xTypes.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/APInt.h"

#include "lib/Dialect/Poly10x/Poly10xCanonicalize.cpp.inc"

namespace mlir {
namespace dummy {
namespace poly10x {

using llvm::APInt;

OpFoldResult ConstantOp::fold(ConstantOp::FoldAdaptor adaptor) {
    return adaptor.getCoefficients();
}

OpFoldResult AddOp::fold(AddOp::FoldAdaptor adaptor) {
    return constFoldBinaryOp<IntegerAttr, APInt, void>(
        adaptor.getOperands(), [&](APInt a, APInt b) { return a + b; });
}

OpFoldResult SubOp::fold(SubOp::FoldAdaptor adaptor) {
    return constFoldBinaryOp<IntegerAttr, APInt, void>(
        adaptor.getOperands(), [&](APInt a, APInt b) { return a - b; });
}

OpFoldResult MulOp::fold(MulOp::FoldAdaptor adaptor) {
    auto lhs = dyn_cast_or_null<DenseElementsAttr>(adaptor.getOperands()[0]);
    auto rhs = dyn_cast_or_null<DenseElementsAttr>(adaptor.getOperands()[1]);

    if (!lhs || !rhs) {
        return nullptr;
    }

    auto degree = cast<PolynomialType>(getResult().getType()).getDegreeBound();
    auto maxIndex = lhs.size() + rhs.size() - 1;

    SmallVector<APInt, 8> result;
    result.reserve(maxIndex);

    for (int i = 0; i < maxIndex; ++i) {
        result.push_back(
            APInt(lhs.getType().getElementType().getIntOrFloatBitWidth(), 0));
    }

    int i = 0;
    for (auto lhsIt = lhs.value_begin<APInt>(); lhsIt != lhs.value_end<APInt>();
         ++lhsIt) {
        int j = 0;
        for (auto rhsIt = rhs.value_begin<APInt>();
             rhsIt != rhs.value_end<APInt>(); ++rhsIt) {
            result[(i + j) % degree] += *rhsIt * (*lhsIt);
            ++j;
        }
        ++i;
    }
    return DenseIntElementsAttr::get(
        RankedTensorType::get(static_cast<int64_t>(result.size()),
                              IntegerType::get(getContext(), 32)),
        result);
}

OpFoldResult FromTensorOp::fold(FromTensorOp::FoldAdaptor adaptor) {
    // Returns null if the cast failed, which corresponds to a failed fold.
    return dyn_cast<DenseIntElementsAttr>(adaptor.getInput());
}

LogicalResult EvalOp::verify() {
    return getPoint().getType().isSignlessInteger(32)
               ? success()
               : emitOpError("argument point must be a 32-bit integer");
}

void AddOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results,
    ::mlir::MLIRContext *context) {}

void SubOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results,
                                        ::mlir::MLIRContext *context) {
    results.add<DifferenceOfSquares>(context);
}

void MulOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results,
    ::mlir::MLIRContext *context) {}

} // namespace poly10x
} // namespace dummy
} // namespace mlir