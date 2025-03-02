#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "lib/Transform/Affine/Passes.h"
#include "lib/Transform/Arith/MulToAdd.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);

  mlir::tutorial::registerAffinePasses();
  mlir::PassRegistration<mlir::dummy::MulToAddPass>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Dummy Pass Driver", registry));
}