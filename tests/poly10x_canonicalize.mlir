// RUN: dummy-opt --canonicalize %s | FileCheck %s

// CHECK-LABEL: test_simple
func.func @test_simple() -> !poly10x.poly<10> {
  // CHECK: poly10x.constant dense<[2, 4, 6]>
  // CHECK-NEXT: return
  %0 = arith.constant dense<[1, 2, 3]> : tensor<3xi32>
  %p0 = poly10x.from_tensor %0 : tensor<3xi32> -> !poly10x.poly<10>
  %2 = poly10x.add %p0, %p0 : (!poly10x.poly<10>, !poly10x.poly<10>) -> !poly10x.poly<10>
  %3 = poly10x.mul %p0, %p0 : (!poly10x.poly<10>, !poly10x.poly<10>) -> !poly10x.poly<10>
  %4 = poly10x.add %2, %3 : (!poly10x.poly<10>, !poly10x.poly<10>) -> !poly10x.poly<10>
  return %2 : !poly10x.poly<10>
}

// CHECK-LABEL: func.func @test_difference_of_squares
// CHECK-SAME: %[[x:.+]]: !poly10x.poly<3>,
// CHECK-SAME: %[[y:.+]]: !poly10x.poly<3>
func.func @test_difference_of_squares(
    %0: !poly10x.poly<3>, %1: !poly10x.poly<3>) -> !poly10x.poly<3> {
  // CHECK: %[[sum:.+]] = poly10x.add %[[x]], %[[y]]
  // CHECK: %[[diff:.+]] = poly10x.sub %[[x]], %[[y]]
  // CHECK: %[[mul:.+]] = poly10x.mul %[[sum]], %[[diff]]
  %2 = poly10x.mul %0, %0 : (!poly10x.poly<3>, !poly10x.poly<3>) -> !poly10x.poly<3>
  %3 = poly10x.mul %1, %1 : (!poly10x.poly<3>, !poly10x.poly<3>) -> !poly10x.poly<3>
  %4 = poly10x.sub %2, %3 : (!poly10x.poly<3>, !poly10x.poly<3>) -> !poly10x.poly<3>
  %5 = poly10x.add %4, %4 : (!poly10x.poly<3>, !poly10x.poly<3>) -> !poly10x.poly<3>
  return %5 : !poly10x.poly<3>
}

// CHECK-LABEL: func.func @test_difference_of_squares_other_uses
// CHECK-SAME: %[[x:.+]]: !poly10x.poly<3>,
// CHECK-SAME: %[[y:.+]]: !poly10x.poly<3>
func.func @test_difference_of_squares_other_uses(
    %0: !poly10x.poly<3>, %1: !poly10x.poly<3>) -> !poly10x.poly<3> {
  // The canonicalization does not occur because x_squared has a second use.
  // CHECK: %[[x_squared:.+]] = poly10x.mul %[[x]], %[[x]]
  // CHECK: %[[y_squared:.+]] = poly10x.mul %[[y]], %[[y]]
  // CHECK: %[[diff:.+]] = poly10x.sub %[[x_squared]], %[[y_squared]]
  // CHECK: %[[sum:.+]] = poly10x.add %[[diff]], %[[x_squared]]
  %2 = poly10x.mul %0, %0 : (!poly10x.poly<3>, !poly10x.poly<3>) -> !poly10x.poly<3>
  %3 = poly10x.mul %1, %1 : (!poly10x.poly<3>, !poly10x.poly<3>) -> !poly10x.poly<3>
  %4 = poly10x.sub %2, %3 : (!poly10x.poly<3>, !poly10x.poly<3>) -> !poly10x.poly<3>
  %5 = poly10x.add %4, %2 : (!poly10x.poly<3>, !poly10x.poly<3>) -> !poly10x.poly<3>
  return %5 : !poly10x.poly<3>
}
