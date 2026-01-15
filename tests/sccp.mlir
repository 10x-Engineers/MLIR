// RUN: dummy-opt -pass-pipeline="builtin.module(func.func(sccp))" %s | FileCheck %s

// Note how sscp creates new constants for the computed values,
// though it does not remove the dead code.

// CHECK-LABEL: test_arith_sccp
// CHECK-NEXT: %[[v0:.*]] = arith.constant 63 : i32
// CHECK-NEXT: %[[v1:.*]] = arith.constant 49 : i32
// CHECK-NEXT: %[[v2:.*]] = arith.constant 14 : i32
// CHECK-NEXT: %[[v3:.*]] = arith.constant 8 : i32
// CHECK-NEXT: %[[v4:.*]] = arith.constant 7 : i32
// CHECK-NEXT: return %[[v2]] : i32
func.func @test_arith_sccp() -> i32 {
  %0 = arith.constant 7 : i32
  %1 = arith.constant 8 : i32
  %2 = arith.addi %0, %0 : i32
  %3 = arith.muli %0, %0 : i32
  %4 = arith.addi %2, %3 : i32
  return %2 : i32
}

// CHECK-LABEL: test_poly_sccp
func.func @test_poly_sccp() -> !poly10x.poly<10> {
  %0 = arith.constant dense<[1, 2, 3]> : tensor<3xi32>
  %p0 = poly10x.from_tensor %0 : tensor<3xi32> -> !poly10x.poly<10>
  // CHECK: poly10x.constant dense<[2, 8, 20, 24, 18]>
  // CHECK: poly10x.constant dense<[1, 4, 10, 12, 9]>
  // CHECK: poly10x.constant dense<[1, 2, 3]>
  // CHECK-NOT: poly10x.mul
  // CHECK-NOT: poly10x.add
  %2 = poly10x.mul %p0, %p0 : (!poly10x.poly<10>, !poly10x.poly<10>) -> !poly10x.poly<10>
  %3 = poly10x.mul %p0, %p0 : (!poly10x.poly<10>, !poly10x.poly<10>) -> !poly10x.poly<10>
  %4 = poly10x.add %2, %3 : (!poly10x.poly<10>, !poly10x.poly<10>) -> !poly10x.poly<10>
  return %2 : !poly10x.poly<10>
}
