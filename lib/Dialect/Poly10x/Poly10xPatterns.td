#ifndef LIB_DIALECT_POLY10X_POLYPATTERNS_TD_
#define LIB_DIALECT_POLY10X_POLYPATTERNS_TD_

include "Poly10xOps.td"
include "mlir/IR/PatternBase.td"

def HasOneUse: Constraint<CPred<"$_self.hasOneUse()">, "has one use">;

// Rewrites (x^2 - y^2) as (x+y)(x-y) if x^2 and y^2 have no other uses.
def DifferenceOfSquares : Pattern<
  (Poly10x_SubOp (Poly10x_MulOp:$lhs $x, $x), (Poly10x_MulOp:$rhs $y, $y)),
  [
    (Poly10x_AddOp:$sum $x, $y),
    (Poly10x_SubOp:$diff $x, $y),
    (Poly10x_MulOp:$res $sum, $diff),
  ],
  [(HasOneUse:$lhs), (HasOneUse:$rhs)]
>;

#endif  // LIB_DIALECT_POLY10X_POLYPATTERNS_TD_