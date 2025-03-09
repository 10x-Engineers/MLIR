#ifndef LIB_DIALECT_POLY_POLY10XOPS_TD_
#define LIB_DIALECT_POLY_POLY10XOPS_TD_

include "Poly10x.td"
include "Poly10xTypes.td"

class Poly10x_BinOp<string mnemonic> : Op<Poly10x_Dialect, mnemonic> {
  let arguments = (ins Polynomial:$lhs, Polynomial:$rhs);
  let results = (outs Polynomial:$output);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($output)";
}

def Poly10x_AddOp : Poly10x_BinOp<"add"> {
  let summary = "Addition operation between polynomials.";
}

def Poly10x_SubOp : Poly10x_BinOp<"sub"> {
  let summary = "Subtraction operation between polynomials.";
}

def Poly10x_MulOp : Poly10x_BinOp<"mul"> {
  let summary = "Subtraction operation between polynomials.";
}

#endif  // LIB_DIALECT_POLY_POLY10XOPS_TD_