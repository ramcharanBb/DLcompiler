#include "Compiler/Dialect/nova/NovaOps.h"
#include "Compiler/Dialect/nova/Broadcast.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::nova;

#define GET_OP_CLASSES
#include "Compiler/Dialect/nova/NovaOps.cpp.inc"


LogicalResult AddOp::inferReturnTypes(
    MLIRContext *context,
    std::optional<Location> loc,
    ValueRange operands,
    DictionaryAttr attributes,
    OpaqueProperties properties,
    RegionRange regions,
    llvm::SmallVectorImpl<Type> &inferredReturnTypes) {

  // Check we have exactly 2 operands
  if (operands.size() != 2) {
    if (loc) {
      mlir::emitError(*loc, "nova.add requires exactly 2 operands");
    }
    return failure();
  }
  
  // Get the types of the operands
  auto lhsType = llvm::dyn_cast<TensorType>(operands[0].getType());
  auto rhsType = llvm::dyn_cast<TensorType>(operands[1].getType());
  

  // Get element type for result
  Type elementType = lhsType.getElementType();
  
  // Handle unranked tensors
  if (!lhsType.hasRank() || !rhsType.hasRank()) {
    inferredReturnTypes.push_back(UnrankedTensorType::get(elementType));
    return success();
  }
  
  // Compute broadcasted shape
  auto broadcastedShape = computeBroadcastShape(lhsType.getShape(), rhsType.getShape());
  
  if (!broadcastedShape) {
    if (loc) {
      mlir::emitError(*loc) 
        << "nova.add: incompatible shapes for broadcasting - "
        << lhsType << " and " << rhsType;
    }
    return failure();
  }
  
  // Create result type with broadcasted shape
  inferredReturnTypes.push_back(
    RankedTensorType::get(*broadcastedShape, elementType));
  
  return success();
}
