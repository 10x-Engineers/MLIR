// RUN: dummy-opt %s > %t
// RUN: FileCheck %s < %t

module {
    // CHECK-LABEL: test_type_syntax
    func.func @test_type_syntax(%arg0: !poly10x.poly<7>) -> !poly10x.poly<7> {
        // CHECK: poly10x.poly
        return %arg0 : !poly10x.poly<7>
    }

    // CHECK-LABEL: test_binop_syntax
    func.func @test_binop_syntax(%arg0: !poly10x.poly<10>, %arg1: !poly10x.poly<10>) -> !poly10x.poly<10> {
        // CHECK: poly10x.add
        %0 = poly10x.add %arg0, %arg1 : (!poly10x.poly<10>, !poly10x.poly<10>) -> !poly10x.poly<10>
        
        // CHECK: poly10x.sub
        %1 = poly10x.sub %arg0, %arg1 : (!poly10x.poly<10>, !poly10x.poly<10>) -> !poly10x.poly<10>
        
        // CHECK: poly10x.mul
        %2 = poly10x.mul %arg0, %arg1 : (!poly10x.poly<10>, !poly10x.poly<10>) -> !poly10x.poly<10>
        
        %3 = arith.constant dense<[1, 2, 3]> : tensor<3xi32>
        // CHECK: poly10x.from_tensor
        %4 = poly10x.from_tensor %3 : tensor<3xi32> -> !poly10x.poly<10>

        %5 = arith.constant 7 : i32

        // CHECK: poly10x.eval
        %6 = poly10x.eval %4, %5 : (!poly10x.poly<10>, i32) -> i32

        %7 = tensor.from_elements %arg0, %arg1 : tensor<2x!poly10x.poly<10>>
        
        // CHECK: poly10x.add
        %8 = poly10x.add %7, %7 : (tensor<2x!poly10x.poly<10>>, tensor<2x!poly10x.poly<10>>) -> tensor<2x!poly10x.poly<10>>

        // CHECK: poly10x.constant
        %10 = poly10x.constant dense<[2, 3, 4]> : tensor<3xi32> : !poly10x.poly<10>
        %11 = poly10x.constant dense<[2, 3, 4]> : tensor<3xi8> : !poly10x.poly<10>
        %12 = poly10x.constant dense<"0x020304"> : tensor<3xi8> : !poly10x.poly<10>
        %13 = poly10x.constant dense<4> : tensor<100xi32> : !poly10x.poly<10>

        // CHECK: poly10x.to_tensor
        %14 = poly10x.to_tensor %1 : !poly10x.poly<10> -> tensor<10xi32>

        return %4 : !poly10x.poly<10>
    }
}