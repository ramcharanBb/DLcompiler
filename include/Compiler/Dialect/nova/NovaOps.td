#ifndef NOVA_OPS
#define NOVA_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "NovaDialect.td"

def Nova_ElementType : AnyTypeOf<[I1, I8, I16, I32, I64, F16, BF16, F32, F64]>;
def Nova_Tensor : TensorOf<[Nova_ElementType]>;

def Nova_AddOp : Op<Nova_Dialect, "add", [Pure, InferTypeOpInterface]> {
  let summary = "Elementwise tensor addition";
  let description = [{
    Adds two tensors elementwise. Result type is automatically inferred.
    
    Example:
      %r = nova.add %a, %b : tensor<2x2xf32>, tensor<2x2xf32>
  }];

  let arguments = (ins Nova_Tensor:$lhs, Nova_Tensor:$rhs);
  let results = (outs Nova_Tensor:$result);
  
  // Fix the assembly format to include result type
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type(operands)";

  let extraClassDeclaration = [{
    static ::mlir::LogicalResult inferReturnTypes(
      ::mlir::MLIRContext *context,
      std::optional<::mlir::Location> loc,
      ::mlir::ValueRange operands,
      ::mlir::DictionaryAttr attributes,
      ::mlir::OpaqueProperties properties,
      ::mlir::RegionRange regions,
      ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes);
  }];
}


//----------------------------------MatMul/Dot Product Operation---------------------------------
def Nova_MatMulOp : Op<Nova_Dialect, "matmul", [Pure, InferTypeOpInterface]> {
  let summary = "Tensor dot product or matrix multiplication";
  let description = [{
    Computes the dot product (matrix multiplication) of two tensors.
    The type inference must handle the shape change based on standard matrix rules.
    
    Example:
      %r = nova.matmul %a, %b : tensor<2x3xf32>, tensor<3x4xf32>
      // Result type inferred as: tensor<2x4xf32>
  }];

  let arguments = (ins Nova_Tensor:$lhs, Nova_Tensor:$rhs);
  let results = (outs Nova_Tensor:$result);
  
  // Fix the assembly format to include operand types
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type(operands)";

  let extraClassDeclaration = [{git config pull.rebase false
    // The C++ implementation must contain logic for matrix multiplication 
    // shape inference (e.g., [M x K] * [K x N] -> [M x N]).
    static ::mlir::LogicalResult inferReturnTypes(
      ::mlir::MLIRContext *context,
      std::optional<::mlir::Location> loc,
      ::mlir::ValueRange operands,
      ::mlir::DictionaryAttr attributes,
      ::mlir::OpaqueProperties properties,
      ::mlir::RegionRange regions,
      ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes);
  }];
}

#endif // NOVA_OPS
