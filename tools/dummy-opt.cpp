#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "lib/Transform/Affine/Passes.h"
#include "lib/Transform/Arith/MulToAdd.h"
#include "lib/Transform/Linalg/MatmulToMmt4d.h"

#include "lib/Dialect/Poly10x/Poly10xDialect.h"

int main(int argc, char **argv) {
    mlir::DialectRegistry registry;

    // register all MLIR dialects
    mlir::registerAllDialects(registry);
    // register all MLIR passes
    mlir::registerAllPasses();
    // registering our Poly10xDialect into dummy-opt
    registry.insert<mlir::dummy::poly10x::Poly10xDialect>();

    // register TableGen'd passes
    mlir::dummy::registerAffinePasses();
    // register hand-authored passes
    mlir::PassRegistration<mlir::dummy::MulToAddPass>();
    mlir::PassRegistration<mlir::dummy::MatmulToMmt4dPass>();

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "Dummy Pass Driver", registry));
}