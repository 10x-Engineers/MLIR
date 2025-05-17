// RUN: dummy-opt --poly10x-to-standard %s | FileCheck %s

// CHECK-LABEL: test_lower_add
func.func @test_lower_add(%0 : !poly10x.poly<10>, %1 : !poly10x.poly<10>) -> !poly10x.poly<10> {
  // CHECK: arith.addi
  %2 = poly10x.add %0, %1 : (!poly10x.poly<10>, !poly10x.poly<10>) -> !poly10x.poly<10>
  return %2 : !poly10x.poly<10>
}

// CHECK-LABEL: test_lower_sub
func.func @test_lower_sub(%0 : !poly10x.poly<10>, %1 : !poly10x.poly<10>) -> !poly10x.poly<10> {
  // CHECK: arith.subi
  %2 = poly10x.sub %0, %1 : (!poly10x.poly<10>, !poly10x.poly<10>) -> !poly10x.poly<10>
  return %2 : !poly10x.poly<10>
}