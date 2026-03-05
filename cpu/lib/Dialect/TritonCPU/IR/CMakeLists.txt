add_triton_library(TritonCPUIR
  Dialect.cpp
  Ops.cpp

  DEPENDS
  TritonCPUTableGen
  TritonCPUAttrsIncGen

  LINK_LIBS PUBLIC
  TritonIR
)
