// RUN: dummy-opt %s

module {
    func.func @main(%arg0: !poly10x.poly<7>) -> !poly10x.poly<7> {
        return %arg0 : !poly10x.poly<7>
    }
}