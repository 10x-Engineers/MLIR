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