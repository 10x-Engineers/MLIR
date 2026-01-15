// RUN: dummy-opt -control-flow-sink %s | FileCheck %s

// Test that operations can be sunk.

// CHECK-LABEL: test_simple_sink
func.func @test_simple_sink(%arg0: i1) -> !poly10x.poly<10> {
    %0 = arith.constant dense<[1, 2, 3]> : tensor<3xi32>
    %p0 = poly10x.from_tensor %0 : tensor<3xi32> -> !poly10x.poly<10>
    %1 = arith.constant dense<[9, 8, 16]> : tensor<3xi32>
    %p1 = poly10x.from_tensor %1 : tensor<3xi32> -> !poly10x.poly<10>
    // CHECK-NOT: poly10x.from_tensor
    // CHECK: scf.if
    %4 = scf.if %arg0 -> (!poly10x.poly<10>) {
        // CHECK: poly10x.from_tensor
        %2 = poly10x.mul %p0, %p0 : (!poly10x.poly<10>, !poly10x.poly<10>) -> !poly10x.poly<10>
        scf.yield %2 : !poly10x.poly<10>
        // CHECK: else
    } else {
        // CHECK: poly10x.from_tensor
        %3 = poly10x.mul %p1, %p1 : (!poly10x.poly<10>, !poly10x.poly<10>) -> !poly10x.poly<10>
        scf.yield %3 : !poly10x.poly<10>
    }
    return %4 : !poly10x.poly<10>
}
