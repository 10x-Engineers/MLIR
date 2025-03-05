// RUN: dummy-opt %s

module {
    func.func @main(%arg0: !poly10x.poly) -> !poly10x.poly {
        return %arg0 : !poly10x.poly
    }
}