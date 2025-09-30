/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::nova::NovaDialect)
namespace mlir {
namespace nova {

NovaDialect::NovaDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<NovaDialect>()) {
  
  initialize();
}

NovaDialect::~NovaDialect() = default;

} // namespace nova
} // namespace mlir
