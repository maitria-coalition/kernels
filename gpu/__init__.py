"""GPU/XLA lanes for contraction-row evaluation.

Layout of this package (see ``gpu/README.md`` for the lane law as it
applies here, and ``rowir`` for the provisional row IR every module
reads through):

- ``rowir`` -- the provisional contraction-row IR + descriptor packing
  (+ the ``fits_i64`` / ``fits_f64`` DATERWI routing gates and the
  three-state ``Verdict``)
- ``reference`` -- the definitional pure-Python evaluator (the semantics)
- ``xla_lane`` -- XLA emission via jax.numpy (the fusible-fragment lane;
  s64 only -- no sound emission exists for enclosures)
- ``ffi_lane`` -- the hand CUDA kernel behind a jax.ffi custom call
- ``pallas_lane`` -- Pallas/Mosaic-GPU lane (inline-PTX escape hatch;
  s64 hadamard-accumulate + the directed-rounding interval cores)
- ``interval`` -- elementwise directed-rounding demo + the shared
  IEEE-total directed scalar ops and exact-rational rounding oracle
- ``ivl_reference`` -- interval-contraction reference strata (exact /
  ideal / directed mirror) + the verdict classifier
- ``ivl_ffi_lane`` -- interval-enclosure contraction on device:
  eval / aliased accumulate / on-device three-state check
- ``ivl_host_lane`` -- fesetround hardware-directed host lane (the
  third corner of the parity witness)
"""
