#ifndef LIB_TRANSFORM_AFFINE_PASSES_TD_
#define LIB_TRANSFORM_AFFINE_PASSES_TD_

include "mlir/Pass/PassBase.td"

def AffineFullUnroll : Pass<"affine-full-unroll"> {
    let summary = "Fully unroll all affine loops";
    let description = [{
        Full unroll all affine loops.
    }];
    let dependentDialects = ["mlir::affine::AffineDialect"];
}

#endif // LIB_TRANSFORM_AFFINE_PASSES_TD_ 