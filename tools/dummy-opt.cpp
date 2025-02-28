#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "lib/Transform/Affine/AffineFullUnroll.h"

int main(int argc, char **argv) {
    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);

    mlir::PassRegistration<mlir::dummy::AffineFullUnrollPass>();

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "Dummy Pass Driver", registry)
    );
}