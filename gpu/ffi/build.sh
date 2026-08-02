#!/usr/bin/env bash
# build librowkernel.so — the contraction-row FFI microkernel.
#
# Pipeline of record: OFFLINE nvcc (the engineering commitment on
# JIT-compiled lanes: offline and runtime-JIT pipelines are different
# instruments and demonstrably disagree on optimization behaviour; this
# lane ships what nvcc assembled). The disassembly branch count is
# printed for the receipts — record it beside timings whenever the
# kernel changes.
set -euo pipefail
cd "$(dirname "$0")"
INC="$(python3 -c 'import jax.ffi; print(jax.ffi.include_dir())')"
NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
ARCH=${ARCH:-native}
echo "jax.ffi.include_dir = $INC"
echo "nvcc = $NVCC  arch = $ARCH"
"$NVCC" -std=c++17 -O3 -shared -Xcompiler -fPIC \
  -I"$INC" -arch="$ARCH" -o librowkernel.so rowkernel.cu
echo "built $(pwd)/librowkernel.so"
CUOBJDUMP="$(dirname "$NVCC")/cuobjdump"
if [ -x "$CUOBJDUMP" ]; then
  BRA=$("$CUOBJDUMP" --dump-sass librowkernel.so | grep -c ' BRA ' || true)
  echo "SASS branch count (BRA): $BRA  (loop structure is by design in"
  echo "this kernel; watch this number for regressions, receipts note it)"
fi
