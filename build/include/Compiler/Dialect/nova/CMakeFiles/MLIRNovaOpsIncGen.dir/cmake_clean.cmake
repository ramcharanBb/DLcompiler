file(REMOVE_RECURSE
  "CMakeFiles/MLIRNovaOpsIncGen"
  "NovaDialect.md"
  "NovaOps.cpp.inc"
  "NovaOps.h.inc"
  "NovaOps.md"
  "NovaOpsDialect.cpp.inc"
  "NovaOpsDialect.h.inc"
  "NovaOpsTypes.cpp.inc"
  "NovaOpsTypes.h.inc"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/MLIRNovaOpsIncGen.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
