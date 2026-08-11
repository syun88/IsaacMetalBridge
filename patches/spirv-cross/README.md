# SPIRV-Cross patches

`scripts/build-spirv-cross.sh` pins Khronos SPIRV-Cross commit
`6c09849fe88c48eaed08413aa022aaa136a3a057` and applies the patches in this
directory before building.

`0001-msl-software-fp64-storage.patch` preserves SPIR-V binary64 buffer layout
while exposing the values as `spvDouble` to generated MSL. The host injects the
software IEEE-754 load/store and float-arithmetic implementation. Scalar,
double3, double4, double4x4, numeric `lf` literals, and `PackDouble2x32` are
covered, including the padded physical representation required by the real
Isaac matrix shaders. The host marks whether the selected entry point actually
uses software FP64; the Vulkan ICD enables declaration-only matrix modules and
gates true-use pipelines until a real dispatch validates their outputs. All 14
captured true-use fixtures now pass on Apple M4 and are enabled by default.

`0002-msl-serialized-atomic64-cas.patch` emits a host-provided helper for
64-bit compare-exchange. Metal on Apple M4 has no native 64-bit atomics, so the
host enables this only for kernels without threadgroup/subgroup coordination
and executes their logical invocations serially on one Metal thread. This is
slower than the source Vulkan kernel but preserves compare, update, and
returned-old-value semantics without an incorrect split 32-bit CAS.
