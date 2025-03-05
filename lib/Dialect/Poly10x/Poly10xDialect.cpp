#include "lib/Dialect/Poly10x/Poly10xDialect.h"
#include "lib/Dialect/Poly10x/Poly10xTypes.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"

#include "lib/Dialect/Poly10x/Poly10xDialect.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/Poly10x/Poly10xTypes.cpp.inc"

namespace mlir {
namespace dummy {
namespace poly10x {

void Poly10xDialect::initialize() {
  // This is where we will register types and operations with the dialect
  addTypes<
  #define GET_TYPEDEF_LIST
  #include "lib/Dialect/Poly10x/Poly10xTypes.cpp.inc"
  >();
}

} // namespace poly10x
} // namespace dummy
} // namespace mlir