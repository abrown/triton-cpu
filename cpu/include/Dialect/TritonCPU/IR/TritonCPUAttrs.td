#ifndef TRITONCPU_ATTRS
#define TRITONCPU_ATTRS

include "TritonCPUDialect.td"

def TTG_PersistentLoop : AttrDef<TritonCPU_Dialect, "PersistentLoop"> {
  let mnemonic = "persistent_loop";
  let description = [{
    Attribute to indicate that a loop is persistent (see `MakePersistent` pass).
  }];
}

#endif
