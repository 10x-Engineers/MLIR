#ifndef LIB_DIALECT_POLY_POLY10XOPS_TD_
#define LIB_DIALECT_POLY_POLY10XOPS_TD_

include "Poly10x.td"
include "Poly10xTypes.td"

def Poly10x_AddOp : Op<Poly10x_Dialect, "add"> {
  let summary = "Addition operation between polynomials.";
  let arguments = (ins Polynomial:$lhs, Polynomial:$rhs);
  let results = (outs Polynomial:$output);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($output)";
}

#endif  // LIB_DIALECT_POLY_POLY10XOPS_TD_