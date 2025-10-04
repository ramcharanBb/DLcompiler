#ifndef NOVA_UTILS_BROADCAST_H
#define NOVA_UTILS_BROADCAST_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>
#include <cstdint>

namespace mlir {
namespace nova {

/// Compute broadcasted shape following NumPy broadcasting rules.
/// Returns std::nullopt if shapes are not broadcast-compatible.
/// 
/// Broadcasting Rules:
///   1. Dimensions are aligned from the right (trailing dimensions)
///   2. Two dimensions are compatible if:
///      - They are equal, OR
///      - One of them is 1, OR
///      - One of them doesn't exist (treat as 1)
///   3. Result dimension is the maximum of the two dimensions
///
/// Examples:
///   [2, 3] + [2, 3] → [2, 3]  (same shape)
///   [2, 3] + [3]    → [2, 3]  (broadcast second operand)
///   [3]    + [2, 3] → [2, 3]  (broadcast first operand)
///   [1, 3] + [2, 1] → [2, 3]  (broadcast both)
///   [2, 3] + [2, 4] → nullopt (incompatible - 3 vs 4)

std::optional<llvm::SmallVector<int64_t, 4>> 
computeBroadcastShape(llvm::ArrayRef<int64_t> lhsShape, 
                      llvm::ArrayRef<int64_t> rhsShape);

/// Check if two shapes are broadcast-compatible without computing result.
bool isBroadcastCompatible(llvm::ArrayRef<int64_t> lhsShape,
                           llvm::ArrayRef<int64_t> rhsShape);

}
}
#endif