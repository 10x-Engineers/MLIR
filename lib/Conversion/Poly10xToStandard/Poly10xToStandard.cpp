#include "lib/Conversion/Poly10xToStandard/Poly10xToStandard.h"

#include "lib/Dialect/Poly10x/Poly10xOps.h"
#include "lib/Dialect/Poly10x/Poly10xTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace dummy {
namespace poly10x {

#define GEN_PASS_DEF_POLY10XTOSTANDARD
#include "lib/Conversion/Poly10xToStandard/Poly10xToStandard.h.inc"

class PolyToStandardTypeConverter : public TypeConverter {
    public:
    PolyToStandardTypeConverter(MLIRContext *ctx) {
        // fallback pattern to convert any type to itself
        // this is useful for types that are not explicitly handled by the
        // conversion patterns
        addConversion([](Type type) { return type; });
        
        // conversion pattern for PolynomialType
        addConversion([ctx](PolynomialType type) -> Type {
            int degreeBound = type.getDegreeBound();
            IntegerType elementTy =
                IntegerType::get(ctx, 32, IntegerType::SignednessSemantics::Signless);
            return RankedTensorType::get({degreeBound}, elementTy);
        });
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

        RewritePatternSet patterns(context);

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