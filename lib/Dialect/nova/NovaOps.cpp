#include "Compiler/Dialect/nova/NovaOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::nova;

#define GET_OP_CLASSES
#include "Compiler/Dialect/nova/NovaOps.cpp.inc"

// Helper function to compute broadcasted shape
static std::optional<SmallVector<int64_t>> 
computeBroadcastShape(ArrayRef<int64_t> lhsShape, ArrayRef<int64_t> rhsShape) {
  SmallVector<int64_t> resultShape;
  
  // Reverse iterate (broadcast aligns from the right)
  int lhsIdx = lhsShape.size() - 1;
  int rhsIdx = rhsShape.size() - 1;
  
  while (lhsIdx >= 0 || rhsIdx >= 0) {
    int64_t lhsDim = (lhsIdx >= 0) ? lhsShape[lhsIdx] : 1;
    int64_t rhsDim = (rhsIdx >= 0) ? rhsShape[rhsIdx] : 1;
    
    // Handle dynamic dimensions
    if (ShapedType::isDynamic(lhsDim) || ShapedType::isDynamic(rhsDim)) {
      // If either is dynamic, result is dynamic
      resultShape.push_back(ShapedType::kDynamic);
    } else if (lhsDim == rhsDim) {
      resultShape.push_back(lhsDim);
    } else if (lhsDim == 1) {
      resultShape.push_back(rhsDim);
    } else if (rhsDim == 1) {
      resultShape.push_back(lhsDim);
    } else {
      // Incompatible shapes
      return std::nullopt;
    }
    
    lhsIdx--;
    rhsIdx--;
  }
  
  // Reverse back to normal order
  std::reverse(resultShape.begin(), resultShape.end());
  return resultShape;
}

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
  
  // Verify both operands are tensors
  if (!lhsType || !rhsType) {
    if (loc) {
      mlir::emitError(*loc, "nova.add operands must be tensors");
    }
    return failure();
  }
  
  // Check element types are compatible
  if (lhsType.getElementType() != rhsType.getElementType()) {
    if (loc) {
      mlir::emitError(*loc, "nova.add operands must have the same element type");
    }
    return failure();
  }
  
  // Get element type for result
  Type elementType = lhsType.getElementType();
  
  // Handle unranked tensors
  if (!lhsType.hasRank() || !rhsType.hasRank()) {
    // If either operand is unranked, result is unranked
    inferredReturnTypes.push_back(UnrankedTensorType::get(elementType));
    return success();
  }
  
  // Compute broadcasted shape
  auto broadcastedShape = computeBroadcastShape(lhsType.getShape(), 
                                                 rhsType.getShape());
  
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

// Optional: Add a verify method for additional runtime checks
// LogicalResult AddOp::verify() {
//   auto lhsType = getLhs().getType().cast<TensorType>();
//   auto rhsType = getRhs().getType().cast<TensorType>();
//   auto resultType = getResult().getType().cast<TensorType>();
  
//   // Verify element types match
//   if (lhsType.getElementType() != rhsType.getElementType() ||
//       lhsType.getElementType() != resultType.getElementType()) {
//     return emitOpError("element types must match across all operands and result");
//   }
  
//   // If all types are ranked, verify the result shape is correct
//   if (lhsType.hasRank() && rhsType.hasRank() && resultType.hasRank()) {
//     auto expectedShape = computeBroadcastShape(lhsType.getShape(), 
//                                                 rhsType.getShape());
//     if (!expectedShape) {
//       return emitOpError("operand shapes are not broadcast-compatible");
//     }
    
//     if (resultType.getShape() != *expectedShape) {
//       return emitOpError("result shape does not match expected broadcast shape");
//     }
//   }
  
//   return success();
// }