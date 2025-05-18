#ifndef LIB_CONVERSION_POLY10XTOSTANDARD_POLY10XTOSTANDARD_TD_
#define LIB_CONVERSION_POLY10XTOSTANDARD_POLY10XTOSTANDARD_TD_

include "mlir/Pass/PassBase.td"

def Poly10xToStandard : Pass<"poly10x-to-standard"> {
    let summary = "Lower `poly10x` to standard MLIR dialects.";

    let description = [{
        This pass lowers the `poly10x` dialect to standard MLIR, a mixture of affine,
        tensor and arith.
    }];

    let dependentDialects = [
        "mlir::arith::ArithDialect",
        "mlir::dummy::poly10x::Poly10xDialect",
        "mlir::tensor::TensorDialect",
        "mlir::scf::SCFDialect",
    ];
}

#endif  // LIB_CONVERSION_POLY10XTOSTANDARD_POLY10XTOSTANDARD_TD_