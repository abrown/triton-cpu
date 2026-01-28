#include "Allocation.h"

#include "cpu/include/TritonCPUToLLVM/Passes.h"

#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/AllocateSharedMemoryUtility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/Support/Alignment.h"

#include "TargetInfo.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_ALLOCATESHAREDMEMORYCPU
#include "cpu/include/TritonCPUToLLVM/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

namespace mlir::triton::cpu {

std::function<unsigned(Operation *)>
getCPUAllocationAnalysisScratchSize(TargetInfo &targetInfo) {
  auto allocation = [&targetInfo](Operation *op) -> unsigned {
    // Handle operations that legitimately need scratch memory on CPU
    if (auto reduceOp = dyn_cast<ReduceOp>(op)) {
      ReduceOpHelper helper(reduceOp);
      return llvm::alignTo(helper.getScratchSizeInBytes(), 64);
    }

    if (auto scanOp = dyn_cast<ScanOp>(op)) {
      ScanLoweringHelper helper(scanOp);
      return llvm::alignTo(helper.getScratchSizeInBytes(), 64);
    }

    if (auto gatherOp = dyn_cast<GatherOp>(op)) {
      GatherLoweringHelper helper(gatherOp);
      return llvm::alignTo(helper.getScratchSizeInBytes(), 64);
    }

    // CPU doesn't need shared memory for layout conversions
    // Just use local memory/registers instead
    if (auto cvtLayout = dyn_cast<gpu::ConvertLayoutOp>(op)) {
      // Return 0 - CPU doesn't use shared memory for layout conversions
      // The conversion happens via vector operations in registers
      return 0;
    }

    // Handle atomic operations if they need scratch space
    if (isa<AtomicRMWOp, AtomicCASOp>(op)) {
      auto value = op->getOperand(0);
      auto smemShape = getRepShapeForAtomic(op->getResult(0));
      auto elems = getNumScratchElements(smemShape);
      if (elems == 0)
        return 0;
      auto elemTy = getElementTypeOrSelf(getPointeeType(value.getType()));
      return llvm::alignTo(
          elems * std::max<int>(8, elemTy.getIntOrFloatBitWidth()) / 8, 64);
    }

    // CPU doesn't have histogram or TMA operations
    // If they appear, return minimal scratch
    if (isa<HistogramOp>(op)) {
      return 0; // CPU doesn't support histogram op
    }

    // No scratch memory needed for this operation
    return 0;
  };

  return allocation;
}

} // namespace mlir::triton::cpu

namespace {

struct AllocateSharedMemoryCPU
    : public mlir::triton::cpu::impl::AllocateSharedMemoryCPUBase<
          AllocateSharedMemoryCPU> {
  using AllocateSharedMemoryCPUBase::AllocateSharedMemoryCPUBase;

  AllocateSharedMemoryCPU() : AllocateSharedMemoryCPUBase() {}

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    mlir::triton::cpu::TargetInfo targetInfo;
    ModuleAllocation allocation(
        mod,
        mlir::triton::cpu::getCPUAllocationAnalysisScratchSize(targetInfo));
    mlir::triton::gpu::attachAllocationSizeAndOffsetAttr(mod, allocation);
  }
};

} // namespace

namespace mlir::triton::cpu {
std::unique_ptr<OperationPass<ModuleOp>> createAllocateSharedMemoryPass() {
  return std::make_unique<AllocateSharedMemoryCPU>();
}
} // namespace mlir::triton::cpu
