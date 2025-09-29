#ifndef NOVA_DIALECT
#define NOVA_DIALECT

include "mlir/IR/DialectBase.td"

//===----------------------------------------------------------------------===//
// Nova Dialect
//===----------------------------------------------------------------------===//


def Nova_Dialect : Dialect {
  let name = "nova";
  let summary = "A high-level dialect for machine learning operations";
  let description = [{
    The Nova dialect provides a high-level representation for ML operations.
  }];
  let cppNamespace = "::mlir::nova";
  let useDefaultTypePrinterParser = 0;  // Changed to 0 to avoid requiring parser methods
  let useDefaultAttributePrinterParser = 0;  // Changed to 0 to avoid requiring parser methods
}

#endif // NOVA_DIALECT
