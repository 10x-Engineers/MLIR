include "mlir/IR/OpBase.td"

def Poly10x_Dialect : Dialect {
    let name = "poly10x";
    let summary = "A dialect for polynomial math";
    let description = [{
        The poly dialect defines types and operations for single-variable
        polynomials over integers.
    }];
    let cppNamespace = "::mlir::dummy::poly10x";
}