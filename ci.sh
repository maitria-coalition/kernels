#!/usr/bin/env bash
# kernels ci.sh
#
# Gates run to completion and report together — a gate that stops
# without reporting the others is withholding information.

set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PASS=(); FAIL=(); NOTES=()
record() { if [ "$2" -eq 0 ]; then PASS+=("$1"); else FAIL+=("$1"); fi; }

# --- gates: workspace hygiene
cargo fmt --check;                        record "fmt" $?
cargo clippy --workspace --all-targets -- -D warnings
                                          record "clippy -D warnings" $?

# --- gates: the conformance battery (ENGINEERING #2), debug AND
# release — lane code must be verdict-identical under both optimizer
# regimes, and the release run is the one that exercises what ships.
cargo test --workspace;                   record "battery (debug)" $?
cargo test --workspace --release;         record "battery (release)" $?

NOTES+=("the battery gates only the lanes this architecture compiles; the other architecture's lanes ride the same battery on their own hardware, receipts under receipts/ (ENGINEERING #3)")
NOTES+=("the GPU/XLA lanes (gpu/) gate on a CUDA box: python -m pytest gpu/tests/ -q there (GPU-lane tests skip loudly off-GPU; reference-only tests run anywhere python+hypothesis exist); receipts under receipts/")

echo
for n in "${PASS[@]:-}"; do [ -n "$n" ] && echo "  PASS  $n"; done
for n in "${FAIL[@]:-}"; do [ -n "$n" ] && echo "  FAIL  $n"; done
for n in "${NOTES[@]:-}"; do [ -n "$n" ] && echo "  NOTE  $n"; done
[ "${#FAIL[@]}" -eq 0 ] && { echo "kernels ci.sh: green."; exit 0; }
echo "kernels ci.sh: ${#FAIL[@]} gate(s) failed."; exit 1
