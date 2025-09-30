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
<<<<<<< HEAD
  let useDefaultTypePrinterParser = 0;
  let useDefaultAttributePrinterParser = 0;
=======
  let useDefaultTypePrinterParser = 0;  // Changed to 0 to avoid requiring parser methods
  let useDefaultAttributePrinterParser = 0;  // Changed to 0 to avoid requiring parser methods
>>>>>>> 08c39335bccfc6f99a9b5e6b29485ac525e18e91
}

#endif // NOVA_DIALECT
