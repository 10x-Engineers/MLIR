// RUN: dummy-opt %s 2>%t; FileCheck %s < %t

func.func @test_invalid_evalop(%arg0: !poly10x.poly<10>, %cst: i64) -> i64 {
  // CHECK: argument point must be a 32-bit integer
  %0 = poly10x.eval %arg0, %cst : (!poly10x.poly<10>, i64) -> i64
  return %0 : i64
}