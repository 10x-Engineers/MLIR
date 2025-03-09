#ifndef LIB_DIALECT_POLY_POLY10XDIALECT_TD_
#define LIB_DIALECT_POLY_POLY10XDIALECT_TD_

include "mlir/IR/OpBase.td"

def Poly10x_Dialect : Dialect {
    let name = "poly10x";
    let summary = "A dialect for polynomial math";
    let description = [{
        The poly dialect defines types and operations for single-variable
        polynomials over integerssss.
    }];
    let cppNamespace = "::mlir::dummy::poly10x";

    let useDefaultTypePrinterParser = 1;
}

#endif  // LIB_DIALECT_POLY_POLY10XDIALECT_TD_