# Rehan Qasim

**AI Compiler Engineer at 10xEngineers.** I make AI models run faster on RISC-V CPUs and custom AI accelerators.

My work sits between the model and the silicon. Hand-written GEMM, GEMV and attention kernels,
the compiler and firmware infrastructure around them, and the profiling tooling needed to know
whether any of it actually helped. Most of it lands in [llama.cpp](https://github.com/ggml-org/llama.cpp),
either for the RISC-V Vector extension or for the Esperanto ET-SoC-1 manycore accelerator.

Before this I designed hardware. My undergraduate thesis was a 3-stage pipelined RV32IMZicsr core
with a loosely coupled 16x16 systolic-array GEMM coprocessor in SystemVerilog on an Artix-7 FPGA,
142x faster than the scalar core on a 60x60 GEMM. That is still the lens I bring to software work,
measure the machine first, then write the kernel it wants.

- Speaking at [RISC-V Summit Europe 2026](https://cfp.riscv-europe.org/eu-summit-2026/talk/TX9SGW/)
  on *Optimizing Llama.cpp and GGML for RISC-V Vector (RVV)*, with Taimur Ahmad
- Interested in LLM inference, compilers for AI accelerators, computer architecture and
  hardware-software co-design
- BS Electrical Engineering, UET Lahore, rank 1 of 200, six gold medals
- [LinkedIn](https://www.linkedin.com/in/rehan-qasim-bhatti) and [email](mailto:rehanbhatti0317@gmail.com)

## Contributions

Each entry links the merged commit carrying my authorship, with the pull request in brackets.
Pull requests opened from my work account (`rehan-10xengineer`) commit under this account.

### RISC-V Vector (RVV) CPU backend, RISE project

RISE RFP RP-014, *Optimizing Llama.cpp and GGML for RVV*, a two person effort with
[Taimur Ahmad](https://github.com/taimur-10x). Merged upstream in `ggml-org/llama.cpp`:

- [`3c7450cee`](https://github.com/ggml-org/llama.cpp/commit/3c7450cee1335eef6f8091fa0498e875249e5595) extend RVV quantization vec dot to higher VLENs ([#22754](https://github.com/ggml-org/llama.cpp/pull/22754)), author
- [`1e796eb41`](https://github.com/ggml-org/llama.cpp/commit/1e796eb41fb51950ada45811a303e57a5f4ea974) 128-bit RVV implementation for quantization vec dot ([#20633](https://github.com/ggml-org/llama.cpp/pull/20633)), author
- [`563753651`](https://github.com/ggml-org/llama.cpp/commit/5637536517ae4ed3eaa22b39c0d479e049097a9b) simd_gemm kernel for the RISC-V Vector extension, used in tiled flash-attention ([#20627](https://github.com/ggml-org/llama.cpp/pull/20627)), author
- [`fbaa95bc2`](https://github.com/ggml-org/llama.cpp/commit/fbaa95bc29b9005586b76ab3d74f485eec05f282) RVV vec dot kernels for quantization types ([#18859](https://github.com/ggml-org/llama.cpp/pull/18859)), author
- [`af237f302`](https://github.com/ggml-org/llama.cpp/commit/af237f3026cecd51b1c6f5e44a4c7cbd747bfde4) RVV repack GEMM and GEMV for quantization types ([#19121](https://github.com/ggml-org/llama.cpp/pull/19121)), co-author
- [`b908baf18`](https://github.com/ggml-org/llama.cpp/commit/b908baf1825b1a89afef87b09e22c32af2ca6548) RVV vec dot kernels for quantization types ([#18784](https://github.com/ggml-org/llama.cpp/pull/18784)), co-author
- [`d34d5ca1e`](https://github.com/ggml-org/llama.cpp/commit/d34d5ca1e9d06d18382feb0cfb6d9d105c86272d) RVV support for llamafile sgemm kernels ([#18199](https://github.com/ggml-org/llama.cpp/pull/18199)), co-author
- [`f716588e6`](https://github.com/ggml-org/llama.cpp/commit/f716588e63224b2f33bb5d13b397fbcfabefa888) extended support for RVV floating-point kernels ([#17318](https://github.com/ggml-org/llama.cpp/pull/17318)), co-author

Merged in the [RISE fork](https://github.com/riseproject-dev/llama.cpp) while upstream review continues:

- [`d19cdcfac`](https://github.com/riseproject-dev/llama.cpp/commit/d19cdcfac) extend RVV GEMM and GEMV to other VLENs ([fork #8](https://github.com/riseproject-dev/llama.cpp/pull/8), upstream [#20723](https://github.com/ggml-org/llama.cpp/pull/20723)), co-author
- [`cd8ed55a7`](https://github.com/riseproject-dev/llama.cpp/commit/cd8ed55a7) tests and performance benchmarks for the floating-point kernels, with the CMake wiring to build them ([fork #2](https://github.com/riseproject-dev/llama.cpp/pull/2)), author

I also wrote the founding commit of [llama.cpp-validation](https://github.com/riseproject-dev/llama.cpp-validation),
the RISC-V cross-compilation toolchain and the initial test and benchmark suite for these kernels.

### Esperanto ET-SoC-1 backend (ggml-et)

Matrix-engine GEMM and GEMV kernels for a 1088-core RISC-V accelerator, across quantized and
floating-point formats, reworking both the prefill and decode paths.

- [`082b326fc`](https://github.com/ggml-org/llama.cpp/commit/082b326fc76f6e9bbb835b3920a3022bfdb6691c) initial ET backend ([#24179](https://github.com/ggml-org/llama.cpp/pull/24179)), co-author
- [`234272cb8`](https://github.com/aifoundry-org/llama.cpp/commit/234272cb8) Q8_0 matrix-engine GEMM kernel for prefill, 6.15x to 9.54x prefill over the vector-unit path ([fork #25](https://github.com/aifoundry-org/llama.cpp/pull/25)), author

Open upstream, measured on Llama-3.2-1B-Instruct against the vector-unit path:

- [#26323](https://github.com/ggml-org/llama.cpp/pull/26323) Q4_0 matrix-engine GEMM for prefill, 3.10x to 3.69x
- [#26326](https://github.com/ggml-org/llama.cpp/pull/26326) Q8_0 matrix-engine GEMM for prefill, 6.15x to 9.54x
- [#26327](https://github.com/ggml-org/llama.cpp/pull/26327) F16 GEMM and GEMV, 7.41x to 16.47x prefill and 2.23x decode
- [#26328](https://github.com/ggml-org/llama.cpp/pull/26328) F32 GEMM and GEMV, 1.13x to 1.52x prefill and 1.92x decode

## Tools

C/C++, Python, RISC-V Assembly, SystemVerilog, MLIR, IREE, RVV intrinsics, PyTorch, llama.cpp, CMake, Perf, GDB, Vivado, QEMU
