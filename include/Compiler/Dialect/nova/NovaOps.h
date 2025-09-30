#ifndef NOVA_OPS_H
#define NOVA_OPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
<<<<<<< HEAD
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
=======
>>>>>>> 08c39335bccfc6f99a9b5e6b29485ac525e18e91

#include "Compiler/Dialect/nova/NovaDialect.h"

#define GET_OP_CLASSES
#include "Compiler/Dialect/nova/NovaOps.h.inc"

#endif // NOVA_OPS_H
