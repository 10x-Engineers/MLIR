// RUN: dummy-opt %s
// RUN FileCheck %s < %t

module {
    // CHECK-LABEL: test_type_syntax
    func.func @test_type_syntax(%arg0: !poly10x.poly<7>) -> !poly10x.poly<7> {
        // CHECK: poly10x.poly
        return %arg0 : !poly10x.poly<7>
    }

    // CHECK-LABEL: test_add_syntax
    func.func @test_add_syntax(%arg0: !poly10x.poly<10>, %arg1: !poly10x.poly<10>) -> !poly10x.poly<10> {
        // CHECK: poly.add
        %0 = poly10x.add %arg0, %arg1 : (!poly10x.poly<10>, !poly10x.poly<10>) -> !poly10x.poly<10>
        return %0 : !poly10x.poly<10>
    }
}