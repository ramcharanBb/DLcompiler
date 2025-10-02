file(REMOVE_RECURSE
  "libMLIRNova.a"
  "libMLIRNova.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/MLIRNova.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
