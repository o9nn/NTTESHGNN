-- Torch7-side producer for the F32 QKV fiber boundary (run with `th`).
-- This is a projection-level experiment, not a GGUF or LLaMA/SAM converter.
local torch = require 'torch'
local nn = require 'nn'

local outdir, d_arg, h_arg, t_arg, checkpoint = arg[1], arg[2], arg[3], arg[4], arg[5]
assert(outdir and d_arg and h_arg and t_arg,
       'usage: th export_torch7_qkv.lua OUT_DIR D H TOKENS [checkpoint.t7]')
local D, H, T = tonumber(d_arg), tonumber(h_arg), tonumber(t_arg)
assert(D and H and T and D > 0 and H > 0 and T > 0 and
       D % 1 == 0 and H % 1 == 0 and T % 1 == 0 and D % H == 0,
       'D, H and TOKENS must be positive integers and D divisible by H')
assert(D <= 1024 and T <= 256, 'prototype bounds: D <= 1024, TOKENS <= 256')

local origin = 'numerical-fixture'
local W, B, X
if checkpoint then
   -- Load only trusted checkpoints: Torch7 deserialization can execute code.
   local obj = torch.load(checkpoint)
   W = assert(obj.weight or (obj.qkv and obj.qkv.weight), 'checkpoint needs weight')
   B = obj.bias or (obj.qkv and obj.qkv.bias)
   X = obj.input
   origin = 'checkpoint'
else
   -- Deterministic numerical fixture, not an AI/agent response or model claim.
   W = torch.FloatTensor(3*D, D)
   B = torch.FloatTensor(3*D)
   for r=1,3*D do
      B[r] = ((r-1) % 7) / 32
      for c=1,D do W[r][c] = (((r-1)*D+c-1) % 17 - 8) / 16 end
   end
end
assert(W:dim() == 2 and W:size(1) == 3*D and W:size(2) == D,
       'expected Torch7 nn.Linear QKV weight [3D,D]')
if B then assert(B:dim() == 1 and B:size(1) == 3*D, 'bias shape must be [3D]')
else B = torch.FloatTensor(3*D):zero() end
if not X then
   X = torch.FloatTensor(T, D)
   for t=1,T do for c=1,D do
      X[t][c] = (((t-1)*D+c-1) % 13 - 6) / 8
   end end
end
assert(X:dim() == 2 and X:size(1) == T and X:size(2) == D,
       'input shape must be [TOKENS,D]')

local source_stride_row, source_stride_col = W:stride(1), W:stride(2)
local source_offset = W:storageOffset() -- Torch7 uses one-based storage offsets.
-- Force independent contiguous FloatStorage: a contiguous view may start at
-- nonzero offset in a larger storage; writing storage() directly would leak it.
W, B, X = W:float():clone():contiguous(), B:float():clone():contiguous(), X:float():clone():contiguous()
assert(W:storageOffset() == 1 and W:storage():size() == W:nElement())
assert(B:storageOffset() == 1 and B:storage():size() == B:nElement())
assert(X:storageOffset() == 1 and X:storage():size() == X:nElement())
local linear = nn.Linear(D, 3*D):float()
linear.weight:copy(W)
linear.bias:copy(B)
local expected = linear:forward(X):float():clone():contiguous()
assert(expected:storageOffset() == 1 and expected:storage():size() == expected:nElement())

local function write_f32(filename, tensor)
   local file = torch.DiskFile(outdir .. '/' .. filename, 'w')
   file:binary()
   file:littleEndianEncoding()
   assert(file:writeFloat(tensor:storage()) == tensor:nElement(), 'short write: ' .. filename)
   file:close()
end
write_f32('weight.f32', W)
write_f32('bias.f32', B)
write_f32('input.f32', X)
write_f32('expected.f32', expected)

local f = assert(io.open(outdir .. '/witness.txt', 'w'))
f:write('schema=fiber-qkv-v1\n',
        'origin=', origin, '\n',
        'dtype=f32-le\n',
        'role_order=QKV\n',
        'source_axes=out,in\n',
        'width=', D, '\nheads=', H, '\ntokens=', T, '\n',
        'source_stride_row=', source_stride_row, '\n',
        'source_stride_col=', source_stride_col, '\n',
        'source_storage_offset_1based=', source_offset, '\n',
        'ggml_weight_ne0=', D, '\nggml_weight_ne1=', 3*D, '\n')
f:close()
print(string.format('Torch7 projected [T=%d,3D=%d]; wrote four F32-LE arrays and witness to %s', T, 3*D, outdir))
