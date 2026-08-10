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
gates only the 13 true-use matrix pipelines until a real dispatch validates
their outputs. Traced and untraced all-enabled Full Warehouse diagnostics
reached the final Metal scene, but did not dispatch those 13 hashes. The one
remaining real module uses 64-bit atomic compare-exchange, which Metal on Apple
M4 does not provide and cannot be emulated correctly as two independent
32-bit atomics.
