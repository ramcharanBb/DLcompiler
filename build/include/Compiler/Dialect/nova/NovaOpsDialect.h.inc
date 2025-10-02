/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: NovaOps.td                                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace nova {

/// A high-level dialect for machine learning operations
/// The Nova dialect provides a high-level representation for ML operations.
class NovaDialect : public ::mlir::Dialect {
  explicit NovaDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~NovaDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("nova");
  }
};
} // namespace nova
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::nova::NovaDialect)
