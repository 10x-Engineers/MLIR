#ifndef LIB_DIALECT_POLY_POLY10XOPS_TD_
#define LIB_DIALECT_POLY_POLY10XOPS_TD_

include "Poly10x.td"
include "Poly10xTypes.td"

include "mlir/Interfaces/SideEffectInterfaces.td"

def Poly10x_ConstantOp : Op<Poly10x_Dialect, "constant", [Pure, ConstantLike]> {
  let summary = "Define a constant polynomial via an attribute.";
  let arguments = (ins AnyIntElementsAttr:$coefficients);
  let results = (outs Polynomial:$output);
  let assemblyFormat = "$coefficients attr-dict `:` type($output)";
  let hasFolder = 1;
}

// Type constraint for poly10x binop arguments: polys, vectors of polys, or
// tensors of polys.
def PolyOrContainer : TypeOrValueSemanticsContainer<Polynomial, "poly-or-container">;

class Poly10x_BinOp<string mnemonic> : Op<Poly10x_Dialect, mnemonic, [Pure, ElementwiseMappable, SameOperandsAndResultElementType]> {
  let arguments = (ins PolyOrContainer:$lhs, PolyOrContainer:$rhs);
  let results = (outs PolyOrContainer:$output);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($output)";
  let hasFolder = 1;
  let hasCanonicalizer = 1;
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

def Poly10x_FromTensorOp : Op<Poly10x_Dialect, "from_tensor", [Pure]> {
  let summary = "Creates a Polynomial from integer coefficients stored in a tensor.";
  let arguments = (ins TensorOf<[AnyInteger]>:$input);
  let results = (outs Polynomial:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
  let hasFolder = 1;
}

def Poly10x_EvalOp : Op<Poly10x_Dialect, "eval"> {
  let summary = "Evaluates a Polynomial at a given input value.";
  let arguments = (ins Polynomial:$input, AnyInteger:$point);
  let results = (outs AnyInteger:$output);
  let assemblyFormat = "$input `,` $point attr-dict `:` `(` type($input) `,` type($point) `)` `->` type($output)";
  let hasVerifier = 1;
}

#endif  // LIB_DIALECT_POLY_POLY10XOPS_TD_