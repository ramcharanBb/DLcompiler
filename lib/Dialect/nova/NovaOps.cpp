#include "Compiler/Dialect/nova/NovaOps.h"
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
    if (loc){
      mlir::emitError(*loc, "nova.add requires exactly 2 operands");
      return failure();
    }
    return failure();
  }
  
  // Get the types of the operands
  auto lhsType = llvm::dyn_cast<TensorType>(operands[0].getType());
  auto rhsType = llvm::dyn_cast<TensorType>(operands[1].getType());
  
  // Verify both operands are tensors
  if (!lhsType || !rhsType) {
    if (loc){
      mlir::emitError(*loc, "nova.add operands must be tensors");
      return failure();
    }
    return failure();
  }
  
  // For now, we assume result type is same as lhs type
  // (Later, when you add broadcasting, this will be more complex)
  inferredReturnTypes.push_back(lhsType);
  
  return success();
}

// ... Matmul operation ...

LogicalResult MatMulOp::inferReturnTypes(
    MLIRContext *context,
    std::optional<Location> loc,
    ValueRange operands,
    DictionaryAttr attributes,
    OpaqueProperties properties,
    RegionRange regions,
    llvm::SmallVectorImpl<Type> &inferredReturnTypes) {
    
  if (operands.size() != 2) {
    if (loc) {
      mlir::emitError(*loc, "nova.matmul requires exactly 2 operands");
      return failure();
    }
    return failure();
  }

  auto lhsType = llvm::dyn_cast<TensorType>(operands[0].getType());
  auto rhsType = llvm::dyn_cast<TensorType>(operands[1].getType());

  if (!lhsType || !rhsType) {
    if (loc) {
      mlir::emitError(*loc, "nova.matmul operands must be tensors");
      return failure();
    }
    return failure();
  }

  auto lhsShape = lhsType.getShape();
  auto rhsShape = rhsType.getShape();

  // 1. VERIFY ELEMENT TYPE CONSISTENCY
  if (lhsType.getElementType() != rhsType.getElementType()) {
      if (loc) {
          mlir::emitError(*loc, "nova.matmul element types must match: ")
              << lhsType.getElementType() << " vs " << rhsType.getElementType();
          return failure();
      }
      return failure();
  }

  // 2. VERIFY RANK AND COMPATIBILITY
  if (lhsShape.size() != 2 || rhsShape.size() != 2) {
    if (loc) {
      mlir::emitError(*loc, "nova.matmul only supports 2D tensors for now (got ranks ")
          << lhsShape.size() << " and " << rhsShape.size() << ")";
      return failure();
    }
    return failure();
  }
  
  // Check for compatible inner dimensions (K)
  if (lhsShape[1] != rhsShape[0]) {
    if (lhsShape[1] != ShapedType::kDynamic && rhsShape[0] != ShapedType::kDynamic) {
        if (loc) {
            mlir::emitError(*loc, "nova.matmul inner dimensions must match: ")
                << lhsShape[1] << " vs " << rhsShape[0];
            return failure();
        }
        return failure();
    }
  }

  // 3. INFER AND ADD RESULT TYPE
  SmallVector<int64_t, 2> resultShape;
  resultShape.push_back(lhsShape[0]); // M
  resultShape.push_back(rhsShape[1]); // N

  auto resultType = RankedTensorType::get(resultShape, lhsType.getElementType());
  
  inferredReturnTypes.push_back(resultType);
  
  return success();
}