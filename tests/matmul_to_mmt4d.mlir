// RUN: dummy-opt --matmul-to-mmt4d %s | FileCheck %s

// CHECK-LABEL: matmul_f32
func.func @matmul_f32(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>, %acc: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %result = linalg.matmul ins(%lhs, %rhs: tensor<?x?xf32>, tensor<?x?xf32>) outs(%acc: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %result: tensor<?x?xf32>
}

// CHECK: linalg.pack
// CHECK: linalg.pack
// CHECK: linalg.pack
// CHECK-NOT: linalg.matmul
// CHECK: linalg.mmt4d
// CHECK: linalg.unpack
