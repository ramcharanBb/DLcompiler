#ifndef NOVA_OPS
#define NOVA_OPS

include "mlir/IR/OpBase.td"
include "NovaDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Nova_ElementType : AnyTypeOf<[I1, I8, I16, I32, I64, F16, BF16, F32, F64, Complex<F32>, Complex<F64>]>;
def Nova_Tensor : TensorOf<[Nova_ElementType]>;
//--------------------------------add---------------------------------------------
def Nova_AddOp : Op<Nova_Dialect, "add", [Pure, InferTypeOpInterface, Commutative, 
  Elementwise, NoMemoryEffect, SameOperandsAndResultElementType]> {
  let summary = "Elementwise tensor addition";
  let description = [{
    Adds two tensors elementwise. Result type is automatically inferred.
    
    Example:
      %r = nova.add %a, %b : tensor<2x2xf32>, tensor<2x2xf32>
  }];

  let arguments = (ins Nova_Tensor:$lhs, Nova_Tensor:$rhs);
  let results = (outs Nova_Tensor:$result);
  
  // Fix the assembly format
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs)";

  //let hasVerifier = 1;

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

//-------------------------------base class for unary------------------------
class Nova_UnaryOp<string mnemonic, list<Trait> traits = [],
                   TypeConstraint typeConstraint = Nova_Tensor> :
    Op<Nova_Dialect, mnemonic, traits> {
  
  let arguments = (ins Nova_Tensor:$operand);
  let results = (outs Nova_Tensor:$result);
  
  let assemblyFormat = "$operand attr-dict `:` type($operand) `->` type($result)";
}
// ----------------sqrt -----------------------------
def Nova_SqrtOp : Nova_UnaryOp<"sqrt"> {
  let summary = "Square root operation";
  let description = [{
    Performs element-wise square root of `operand` tensor.
    Example:
    %result = nova.sqrt%operand : tensor<5xf32>
  }];
}
// ----------------neg-----------------------------
def Nova_NegOp : Nova_UnaryOp<"neg"> {
  let summary = "Elementwise negation";
  let description = [{
    Negates each element of the tensor.
    
    Example:
      %r = nova.neg %a : tensor<2x3xf32>
  }];
}
//---------------------abs----------------------
def Nova_AbsOp : Nova_UnaryOp<"abs"> {
  let summary = "Elementwise absolute value";
   let description = [{
    returns the absolute value of a tensor
    
    Example:
      %r = nova.abs %a : tensor<2x3xf32>
  }];
}
//---------------------exp----------------------
def Nova_ExpOp : Nova_UnaryOp<"exp"> {
  let summary = "Elementwise exponential";
   let description = [{
    returns the exponential value of a tensor
    
    Example:
      %r = nova.exp %a : tensor<2x3xf32>
  }];
}
//---------------------log----------------------
def Nova_LogOp : Nova_UnaryOp<"log"> {
  let summary = "Elementwise natural logarithm";
   let description = [{
    returns the logarithm value of a tensor    
    Example:
      %r = nova.log %a : tensor<2x3xf32>
  }];
}

//---------------------tanh----------------------
def Nova_TanhOp : Nova_UnaryOp<"tanh"> {
  let summary = "Elementwise hyperbolic tangent";
   let description = [{
    elementwise tangent
    Example:
      %r = nova.tanh %a : tensor<2x3xf32>
  }];
}

//---------------------relu----------------------
def Nova_ReluOp : Nova_UnaryOp<"relu"> {
  let summary = "Elementwise ReLU activation";
  let description = [{
    Applies ReLU: max(0, x)
    Example:
      %r = nova.relu %a : tensor<2x3xf32>
  }];
}

//---------------------ceil----------------------
def Nova_CeilOp: Nova_UnaryOp<"ceil">{
  let summary = "Ceil operation";
  let description = [{
    Performs element-wise ceil of `operand` tensor and produces a `result` tensor.
    Example:
    %result = nova.ceil %operand : tensor<5xf32>
  }];
}

//----------------------floor--------------------
def Nova_FloorOp: Nova_UnaryOp<"floor">{
  let summary = "Ceil operation";
  let description = [{
    Performs element-wise ceil of `operand` tensor and produces a `result` tensor.
    Example:
    %result = nova.floor %operand : tensor<5xf32>
  }];
}


#endif // NOVA_OPS
