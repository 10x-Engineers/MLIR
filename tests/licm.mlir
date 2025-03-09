// RUN: dummy-opt %s --loop-invariant-code-motion > %t
// RUN: FileCheck %s < %t

module {
  // CHECK-LABEL: test_loop_invariant_code_motion
  func.func @test_loop_invariant_code_motion() -> !poly10x.poly<10> {
    %0 = arith.constant dense<[1, 2, 3]> : tensor<3xi32>
    %p0 = poly10x.from_tensor %0 : tensor<3xi32> -> !poly10x.poly<10>

    %1 = arith.constant dense<[9, 8, 16]> : tensor<3xi32>
    %p1 = poly10x.from_tensor %1 : tensor<3xi32> -> !poly10x.poly<10>
    // CHECK: poly10x.mul

    // CHECK: affine.for
    %ret_val = affine.for %i = 0 to 100 iter_args(%sum_iter = %p0) -> !poly10x.poly<10> {
      // The poly10x.mul should be hoisted out of the loop.
      // CHECK-NOT: poly10x.mul
      %2 = poly10x.mul %p0, %p1 : (!poly10x.poly<10>, !poly10x.poly<10>) -> !poly10x.poly<10>
      %sum_next = poly10x.add %sum_iter, %2 : (!poly10x.poly<10>, !poly10x.poly<10>) -> !poly10x.poly<10>
      affine.yield %sum_next : !poly10x.poly<10>
    }

    return %ret_val : !poly10x.poly<10>
  }
}