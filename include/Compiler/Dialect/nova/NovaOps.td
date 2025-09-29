#ifndef NOVA_OPS
#define NOVA_OPS

include "mlir/IR/OpBase.td"
include "NovaDialect.td"

def Nova_ElementType : AnyTypeOf<[I1, I8, I16, I32, I64, F16, BF16, F32, F64]>;
def Nova_Tensor : TensorOf<[Nova_ElementType]>;

def Nova_AddOp : Op<Nova_Dialect, "add"> {
  let summary = "Elementwise tensor addition";
  let description = "Adds two tensors elementwise.";

  let arguments = (ins Nova_Tensor:$lhs, Nova_Tensor:$rhs);
  let results = (outs Nova_Tensor:$result);
  
  // Fix the assembly format to include result type
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type(operands) `->` type($result)";
}

#endif // NOVA_OPS
