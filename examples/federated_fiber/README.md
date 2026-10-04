# Federated QKV fiber boundary — executable experiment

This **projection-level** prototype brings three previously isolated components into one falsifiable frame: native Torch7 `nn.Linear` weight semantics; NTTESHGNN's storage/view materialization; and a real CPU graph in [cogpy/coggml at `1033a16c`](https://github.com/cogpy/coggml/tree/1033a16c2fb00746906c219c7abaee2242ccf1d6). It is **not** a trained model, a GGUF converter, a decoder KV cache, or a CANS solver.

## Contract

| Representation | Logical and physical contract |
|---|---|
| Torch7 source | `W:[3D,D]` ordered Q rows, K rows, V rows; `X:[T,D]`, `b:[3D]`. A checkpoint's original Torch7 strides and one-based storage offset are recorded. |
| Serialized boundary | Independently cloned contiguous F32 tensors as little-endian `weight.f32`, `input.f32`, `bias.f32`, `expected.f32`; strict `witness.txt` lists D, H, T, QKV role order, and source stride evidence. Torch7 computes `expected = nn.Linear(X)`. |
| NTTESHGNN | Wrap `[D,3D]`, reshape `[D,D,3]`, permute using **`dims[output_axis]=input_axis`** to `[3,D,D]`, materialize logical scalars by stride into independent storage. |
| GGML leaf and graph | New, copied F32 leaves `W.ne=[D,3D]`, `X.ne=[D,T]`; `MUL_MAT`, repeated bias, reshape `[d,H,3,T]`, permute to `[3,H,d,T]`, `CONT`, actual CPU compute. The pinned GGML permute arguments instead mean **`axis[input_axis]=output_axis`**. |
| Comparison | Every projected and role/head/channel-indexed scalar must match the independent reference to `1e-4 * max(1,abs(reference))`. This is a numerical parity check, not a model-performance claim. |

Input bounds: `D≤1024`, `T≤256`, `D % H == 0`; F32, little-endian IEEE host, CPU only. The text witness cannot safely encode or prove arbitrary target GGUF tensor layouts. Input file lengths are exact; duplicates and inconsistent metadata are rejected. Source pointer aliasing stops at the boundary.

## Build and run against the pinned GGML source

```bash
gh repo clone cogpy/coggml /tmp/coggml-fiber -- --depth=1
# At the time of the experiment, HEAD = 1033a16c2fb00746906c219c7abaee2242ccf1d6.
test "$(git -C /tmp/coggml-fiber rev-parse HEAD)" = 1033a16c2fb00746906c219c7abaee2242ccf1d6
cmake -S . -B /tmp/ntt-fiber-build -DNT_BUILD_FIBER_GGML_DEMO=ON \
  -DGGML_SOURCE_DIR=/tmp/coggml-fiber -DBUILD_SHARED_LIBS=OFF \
  -DGGML_CUDA=OFF -DGGML_METAL=OFF -DGGML_BUILD_EXAMPLES=OFF -DGGML_BUILD_TESTS=OFF
cmake --build /tmp/ntt-fiber-build -j4
ctest --test-dir /tmp/ntt-fiber-build --output-on-failure
```

The graph test creates **independent numerical fixtures** at `(D,H,T)=(4,2,3),(6,3,2),(9,3,5)`; these are test data, not generated agent responses and not evidence that Torch7 ran. It also tests rejection of an incompatible head count, reversed role order, a truncated binary file, and a corrupted reference output. The C tensor regression tests include the formerly failing `2×3` transposed-to-contiguous case, rank-4 permute/inverse, and an offset slice.

To exercise **native Torch7**, separately install its historical `th`, `torch` and `nn` environment, and provide an output directory:

```bash
mkdir -p /tmp/qkv-from-torch7
th examples/federated_fiber/export_torch7_qkv.lua /tmp/qkv-from-torch7 8 2 3
/tmp/ntt-fiber-build/fiber_qkv_graph /tmp/qkv-from-torch7
# Or pass a trusted checkpoint.t7 as the fifth argument. Its table/module must
# expose weight:[3D,D], optional bias:[3D], optional input:[T,D].
```

**The Torch7 command was not run in this Ubuntu sandbox:** `th`, `torch` and `nn` were unavailable. The Lua file passed `luac -p` (syntax, not native runtime validation). Loading a `.t7` checkpoint is deserialization: use a **trusted** file only. A checkpoint without `input` is compared on a generated numerical activation fixture, not a historical model activation.

## Scope boundaries and next experiment

- Role order QKV and equal Q/K/V width are deliberate hypotheses to test; **do not** reuse for grouped-query attention or arbitrary llama.cpp GGUF weights.
- The owned C library's existing tests initially passed while a targeted transposed-copy probe failed on 4 of 6 values. The fixed path copies each logical scalar using `src->nb[axis]`; it rejects block-quantized/sub-byte and non-CPU materialization rather than guessing their physical representation.
- Torch7 contiguous-last-dimension layout and `DiskFile.writeFloat(FloatStorage)` are documented by [Torch7 Tensor](https://torch7.readthedocs.io/en/latest/tensor/) and [Torch7 File](https://github.com/torch/torch7/blob/master/doc/file.md). Pinned GGML axis behavior is in [`src/ggml.c`](https://github.com/cogpy/coggml/blob/1033a16c2fb00746906c219c7abaee2242ccf1d6/src/ggml.c#L3752-L3803).
- Future `CANS` token geometry requires a chosen mesh/chart/metric/incidence and a separately typed feature result; it is **not** a stride validator. Future decoder cache requires explicit session/sequence APIs; the supplied prompt-only HTTP adapter cannot supply them.

See [`docs/federated_fiber/EXPERIMENT.md`](../../docs/federated_fiber/EXPERIMENT.md) for the seed→experiment→counterexample→refinement ledger and Mermaid contracts.
