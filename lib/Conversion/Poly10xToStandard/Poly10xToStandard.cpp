#include "lib/Conversion/Poly10xToStandard/Poly10xToStandard.h"

#include "lib/Dialect/Poly10x/Poly10xOps.h"
#include "lib/Dialect/Poly10x/Poly10xTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
namespace mlir {
namespace dummy {
namespace poly10x {

#define GEN_PASS_DEF_POLY10XTOSTANDARD
#include "lib/Conversion/Poly10xToStandard/Poly10xToStandard.h.inc"

class Poly10xToStandardTypeConverter : public TypeConverter {
  public:
    Poly10xToStandardTypeConverter(MLIRContext *ctx) {
        // fallback pattern to convert any type to itself
        // this is useful for types that are not explicitly handled by the
        // conversion patterns
        addConversion([](Type type) { return type; });

        // conversion pattern for PolynomialType
        addConversion([ctx](PolynomialType type) -> Type {
            int degreeBound = type.getDegreeBound();
            IntegerType elementTy = IntegerType::get(
                ctx, 32, IntegerType::SignednessSemantics::Signless);
            return RankedTensorType::get({degreeBound}, elementTy);
        });
    }
};

struct ConvertAdd : public OpConversionPattern<AddOp> {
    ConvertAdd(mlir::MLIRContext *context)
        : OpConversionPattern<AddOp>(context) {}

    using OpConversionPattern::OpConversionPattern;

    // OpAdaptor holds type-converted operands during dialect conversion.
    // Uses table-gen names instead of generic getOperand-style access.
    // using OpAdaptor = AddOp::Adaptor;

    // AddOp provides access to original, unconverted operands/results.
    // Same as in standard rewrite patterns.
    LogicalResult
    matchAndRewrite(AddOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {

        arith::AddIOp addOp = rewriter.create<arith::AddIOp>(
            op.getLoc(), adaptor.getLhs(), adaptor.getRhs());
        rewriter.replaceOp(op, addOp);
        return success();
    }
};

struct ConvertSub : public OpConversionPattern<SubOp> {
    ConvertSub(mlir::MLIRContext *context)
        : OpConversionPattern<SubOp>(context) {}

    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(SubOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {

        arith::SubIOp subOp = rewriter.create<arith::SubIOp>(
            op.getLoc(), adaptor.getLhs(), adaptor.getRhs());
        rewriter.replaceOp(op, subOp);
        return success();
    }
};

struct ConvertFromTensor : public OpConversionPattern<FromTensorOp> {
    ConvertFromTensor(mlir::MLIRContext *context)
        : OpConversionPattern<FromTensorOp>(context) {}

    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(FromTensorOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
        auto resultTensorTy = cast<RankedTensorType>(
            typeConverter->convertType(op->getResultTypes()[0]));
        auto resultShape = resultTensorTy.getShape()[0];
        auto resultEltTy = resultTensorTy.getElementType();

        auto inputTensorTy = op.getInput().getType();
        auto inputShape = inputTensorTy.getShape()[0];

        // Zero pad the tensor if the coefficients' size is less than the
        // polynomial degree.

        // ImplicitLocOpBuilder maintains a 'current location', allowing use of
        // the create<> method without specifying the location. It is otherwise
        // the same as OpBuilder.
        ImplicitLocOpBuilder b(op.getLoc(), rewriter);
        auto coeffValue = adaptor.getInput();
        if (inputShape < resultShape) {
            SmallVector<OpFoldResult, 1> low, high;
            low.push_back(rewriter.getIndexAttr(0));
            high.push_back(rewriter.getIndexAttr(resultShape - inputShape));
            coeffValue = b.create<tensor::PadOp>(
                resultTensorTy, coeffValue, low, high,
                b.create<arith::ConstantOp>(
                    rewriter.getIntegerAttr(resultEltTy, 0)),
                /*nofold=*/false);
        }

        rewriter.replaceOp(op, coeffValue);
        return success();
    }
};

struct ConvertConstant : public OpConversionPattern<ConstantOp> {
    ConvertConstant(mlir::MLIRContext *context)
        : OpConversionPattern<ConstantOp>(context) {}

    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(ConstantOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
        ImplicitLocOpBuilder b(op.getLoc(), rewriter);
        auto constOp = b.create<arith::ConstantOp>(adaptor.getCoefficients());
        auto fromTensorOp =
            b.create<FromTensorOp>(op.getResult().getType(), constOp);
        rewriter.replaceOp(op, fromTensorOp.getResult());
        return success();
    }
};

struct ConvertToTensor : public OpConversionPattern<ToTensorOp> {
    ConvertToTensor(mlir::MLIRContext *context)
        : OpConversionPattern<ToTensorOp>(context) {}

    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(ToTensorOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
        rewriter.replaceOp(op, adaptor.getInput());
        return success();
    }
};

struct ConvertMul : public OpConversionPattern<MulOp> {
    ConvertMul(mlir::MLIRContext *context)
        : OpConversionPattern<MulOp>(context) {}

    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(MulOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
        auto polymulTensorType =
            cast<RankedTensorType>(adaptor.getLhs().getType());
        auto numTerms = polymulTensorType.getShape()[0];
        ImplicitLocOpBuilder b(op.getLoc(), rewriter);

        // Create an all-zeros tensor to store the result
        auto polymulResult = b.create<arith::ConstantOp>(
            polymulTensorType, DenseElementsAttr::get(polymulTensorType, 0));

        // Loop bounds and step.
        auto lowerBound =
            b.create<arith::ConstantOp>(b.getIndexType(), b.getIndexAttr(0));
        auto numTermsOp = b.create<arith::ConstantOp>(b.getIndexType(),
                                                      b.getIndexAttr(numTerms));
        auto step =
            b.create<arith::ConstantOp>(b.getIndexType(), b.getIndexAttr(1));

        auto p0 = adaptor.getLhs();
        auto p1 = adaptor.getRhs();

        // for i = 0, ..., N-1
        //   for j = 0, ..., N-1
        //     product[i+j (mod N)] += p0[i] * p1[j]

        // loopState is iter_args
        auto outerLoop = b.create<scf::ForOp>(
            lowerBound, numTermsOp, step, ValueRange(polymulResult.getResult()),
            [&](OpBuilder &builder, Location loc, Value p0Index,
                ValueRange loopState) {
                ImplicitLocOpBuilder b(op.getLoc(), builder);
                auto innerLoop = b.create<scf::ForOp>(
                    lowerBound, numTermsOp, step, loopState,
                    [&](OpBuilder &builder, Location loc, Value p1Index,
                        ValueRange loopState) {
                        ImplicitLocOpBuilder b(op.getLoc(), builder);
                        auto accumTensor = loopState.front();
                        auto destIndex = b.create<arith::RemUIOp>(
                            b.create<arith::AddIOp>(p0Index, p1Index),
                            numTermsOp);
                        auto mulOp = b.create<arith::MulIOp>(
                            b.create<tensor::ExtractOp>(p0,
                                                        ValueRange(p0Index)),
                            b.create<tensor::ExtractOp>(p1,
                                                        ValueRange(p1Index)));
                        auto result = b.create<arith::AddIOp>(
                            mulOp, b.create<tensor::ExtractOp>(
                                       accumTensor, destIndex.getResult()));
                        auto stored = b.create<tensor::InsertOp>(
                            result, accumTensor, destIndex.getResult());
                        b.create<scf::YieldOp>(stored.getResult());
                    });

                b.create<scf::YieldOp>(innerLoop.getResults());
            });

        rewriter.replaceOp(op, outerLoop.getResult(0));
        return success();
    }
};

struct ConvertEval : public OpConversionPattern<EvalOp> {
    ConvertEval(mlir::MLIRContext *context)
        : OpConversionPattern<EvalOp>(context) {}

    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(EvalOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
        auto polyTensorType =
            cast<RankedTensorType>(adaptor.getInput().getType());
        auto numTerms = polyTensorType.getShape()[0];
        ImplicitLocOpBuilder b(op.getLoc(), rewriter);

        auto lowerBound =
            b.create<arith::ConstantOp>(b.getIndexType(), b.getIndexAttr(1));
        auto numTermsOp = b.create<arith::ConstantOp>(
            b.getIndexType(), b.getIndexAttr(numTerms + 1));
        auto step = lowerBound;

        auto poly = adaptor.getInput();
        auto point = adaptor.getPoint();

        // Horner's method:
        //
        // accum = 0
        // for i = 1, 2, ..., N
        //   accum = point * accum + coeff[N - i]
        auto accum =
            b.create<arith::ConstantOp>(b.getI32Type(), b.getI32IntegerAttr(0));
        auto loop = b.create<scf::ForOp>(
            lowerBound, numTermsOp, step, accum.getResult(),
            [&](OpBuilder &builder, Location loc, Value loopIndex,
                ValueRange loopState) {
                ImplicitLocOpBuilder b(op.getLoc(), builder);
                auto accum = loopState.front();
                auto coeffIndex =
                    b.create<arith::SubIOp>(numTermsOp, loopIndex);
                auto mulOp = b.create<arith::MulIOp>(point, accum);
                auto result = b.create<arith::AddIOp>(
                    mulOp,
                    b.create<tensor::ExtractOp>(poly, coeffIndex.getResult()));
                b.create<scf::YieldOp>(result.getResult());
            });

        rewriter.replaceOp(op, loop.getResult(0));
        return success();
    }
};

struct Poly10xToStandard : impl::Poly10xToStandardBase<Poly10xToStandard> {
    using Poly10xToStandardBase::Poly10xToStandardBase;

    void runOnOperation() override {
        MLIRContext *context = &getContext();
        auto *module = getOperation();

        // ConversionTarget is used to specify ops as legal, illegal, or
        // dynamically legal
        ConversionTarget target(*context);
        // specifying that poly10x dialect is illegal and should not exist after
        // this pass runs
        target.addIllegalDialect<Poly10xDialect>();

        // Mark Arith as legal, so that we can use it in the conversion
        target.addLegalDialect<arith::ArithDialect>();

        RewritePatternSet patterns(context);
        Poly10xToStandardTypeConverter typeConverter(context);
        patterns.add<ConvertAdd, ConvertSub, ConvertFromTensor, ConvertToTensor,
                     ConvertMul, ConvertConstant, ConvertEval>(typeConverter,
                                                               context);

        populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
            patterns, typeConverter);
        target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
            return typeConverter.isSignatureLegal(op.getFunctionType()) &&
                   typeConverter.isLegal(&op.getBody());
        });

        populateReturnOpTypeConversionPattern(patterns, typeConverter);
        target.addDynamicallyLegalOp<func::ReturnOp>(
            [&](func::ReturnOp op) { return typeConverter.isLegal(op); });

        populateCallOpTypeConversionPattern(patterns, typeConverter);
        target.addDynamicallyLegalOp<func::CallOp>(
            [&](func::CallOp op) { return typeConverter.isLegal(op); });

        populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
        target.markUnknownOpDynamicallyLegal([&](Operation *op) {
            return isNotBranchOpInterfaceOrReturnLikeOp(op) ||
                   isLegalForBranchOpInterfaceTypeConversionPattern(
                       op, typeConverter) ||
                   isLegalForReturnOpTypeConversionPattern(op, typeConverter);
        });

        // a partial conversion will legalize as many operations to the target
        // as possible, but will allow pre-existing operations that were not
        // explicitly marked as “illegal” to remain unconverted. official docs
        // are good so read up on different kinds of conversions from there
        if (failed(
                applyPartialConversion(module, target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

} // namespace poly10x
} // namespace dummy
} // namespace mlir
