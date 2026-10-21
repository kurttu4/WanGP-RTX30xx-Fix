# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""BF16 convolutions on FP16 tensor cores, for GPUs with FP16 but no BF16 tensor cores (validated on Turing, compute capability 7.5),
where PyTorch has no fast BF16 convolution (cuDNN has no BF16 kernels there: `slow_conv_dilated3d`, i.e. vol2col and a CUDA-core GEMM).

A BF16 value converts exactly to FP16 (8 significant bits into 11) while its magnitude stays in FP16's normal range. The weight is scaled
by a power of two (largest |w| in [0.5, 1)); the input by the power of two that keeps every input below 2^15 and every possible output
below 2^15 (largest |x| times the largest L1 norm of an output channel's weights), so nothing overflows; cuDNN accumulates in FP32; the
FP16 result is scaled back, the bias added in FP32 and rounded once to BF16, in place. Precision: values that end up below FP16's normal
range lose bits, and the result rounds to FP16 (11 bits) before BF16 (8 bits): LTX-2.5's VAE decoder, encoder and spatial upsampler
measured within +-3% of the PyTorch BF16 route's error against FP32, 4.8-6.8x faster. Used with every kernel precision (user decision,
2026-10-21: a few % more error than PyTorch's route is worth it where PyTorch has no fast BF16 convolution), an exception to the strict rule.

install(module, device) at load time: on an eligible GPU, each Conv1d / Conv2d / Conv3d of the module computes its BF16 CUDA calls this
way; other dtypes, devices, autograd, compilation and graph capture keep PyTorch's route. Any model can call it
on its VAE, upsampler or other convolutional component."""
import functools

import torch
import torch.nn.functional as F

from .dispatch import choose, cuda_inference

try:
    import triton
    import triton.language as tl
    from .bf16_bits import bf16_bits_to_fp32, fp32_to_bf16_bits
except ImportError:
    triton = None

_BLOCK = 1024
_SHIFT_LIMIT = 120  # powers of two applied in two halves: exact in FP32 for every shift of input and weight
_CONVOLUTIONS = {torch.nn.Conv1d: F.conv1d, torch.nn.Conv2d: F.conv2d, torch.nn.Conv3d: F.conv3d}


@functools.cache
def _supported(index):
    return torch.cuda.get_device_capability(index) == (7, 5)


def supported(device) -> bool:
    """Validated on compute capability 7.5 only (Volta has the same FP16 tensor cores, untested); needs Triton for the conversions."""
    device = torch.device(device)
    return triton is not None and device.type == "cuda" and torch.version.hip is None and torch.cuda.is_available() and _supported(device.index)


def install(module, device) -> int:
    """Route the BF16 calls of module's Conv1d/2d/3d layers on an eligible GPU (supported(device)); returns the number of layers."""
    if not supported(device):
        return 0
    count = 0
    for layer in module.modules():
        if type(layer) in _CONVOLUTIONS and "_conv_forward" not in layer.__dict__:
            layer._conv_forward = functools.partial(_conv_forward, layer, layer._conv_forward)
            count += 1
    return count


def _conv_forward(layer, original, input, weight, bias):
    if (not cuda_inference(input, sm120=False, inference=False) or weight.dtype != torch.bfloat16
            or type(weight) not in (torch.Tensor, torch.nn.Parameter) or weight.device != input.device
            or (bias is not None and bias.dtype != torch.bfloat16) or torch.cuda.is_current_stream_capturing()):
        return original(input, weight, bias)
    choose("BF16 Convolution", owned_triton=True)
    return convolution(layer, input, weight, bias)


def _weight_shift(layer, weight):
    """(power of two applied to the weight, largest L1 norm of an output channel's scaled weights): measured once per layer, at its first
    call on the GPU (one synchronization)."""
    stats = layer.__dict__.get("_scaled_fp16_stats")
    if stats is None:
        magnitude = weight.abs()
        _, exponent = torch.frexp(magnitude.amax().float())
        l1 = magnitude.float().flatten(1).sum(1).amax()
        shift = max(-_SHIFT_LIMIT, min(_SHIFT_LIMIT, -int(exponent.item())))
        stats = layer._scaled_fp16_stats = (shift, float(l1.item()) * 2.0 ** shift * (1 + 2.0 ** -10))  # bound rounded up
    return stats


def convolution(layer, input, weight, bias):
    """layer's convolution (its stride, padding, padding mode, dilation, groups) of BF16 CUDA input, weight and bias on FP16 tensor cores."""
    weight_shift, l1 = _weight_shift(layer, weight)
    x = input.contiguous()
    low, high = torch.aminmax(x)
    x16 = torch.empty(x.shape, dtype=torch.float16, device=x.device)
    shift = torch.empty(1, dtype=torch.int32, device=x.device)
    _to_fp16_kernel[(triton.cdiv(x.numel(), _BLOCK),)](x.view(torch.int16), x16, x.numel(), 0, low.view(torch.int16), high.view(torch.int16), l1,
                                                        shift, True, _BLOCK)
    x = None
    weight = weight.contiguous()
    w16 = torch.empty(weight.shape, dtype=torch.float16, device=weight.device)
    _to_fp16_kernel[(triton.cdiv(weight.numel(), _BLOCK),)](weight.view(torch.int16), w16, weight.numel(), weight_shift, None, None, 0.0,
                                                             None, False, _BLOCK)
    padding = layer.padding
    if layer.padding_mode != "zeros":
        x16, padding = F.pad(x16, layer._reversed_padding_repeated_twice, mode=layer.padding_mode), 0
    y = _CONVOLUTIONS[type(layer)](x16, w16, None, layer.stride, padding, layer.dilation, layer.groups).contiguous()
    x16 = w16 = None
    channels, spatial = y.shape[1], y[0, 0].numel()
    _to_bf16_kernel[(triton.cdiv(y.numel(), _BLOCK),)](y, y.view(torch.int16), None if bias is None else bias.view(torch.int16), shift,
                                                        weight_shift, y.numel(), channels, spatial, bias is not None, _BLOCK)
    return y.view(torch.bfloat16)


if triton is not None:
    @triton.jit
    def _power_of_two(shift):
        return ((shift + 127) << 23).to(tl.float32, bitcast=True)

    @triton.jit
    def _exponent(x):
        return ((x.to(tl.uint32, bitcast=True) >> 23) & 0xFF).to(tl.int32) - 127

    @triton.jit(do_not_specialize=["n", "fixed_shift", "l1"])
    def _to_fp16_kernel(X, Y, n, fixed_shift, LOW, HIGH, l1, SHIFT, DYNAMIC: tl.constexpr, BLOCK: tl.constexpr):
        # BF16 bits -> FP16 times a power of two: fixed (a weight), or from the input's extremes and the weight's L1 bound (stored in SHIFT)
        offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        if DYNAMIC:
            largest = tl.maximum(tl.abs(bf16_bits_to_fp32(tl.load(LOW))), tl.abs(bf16_bits_to_fp32(tl.load(HIGH))))
            bound = largest * l1
            # 14 - exponent: inputs and outputs below 2^15; zero, infinite and NaN inputs keep a shift of 0
            shift = 14 - tl.maximum(_exponent(bound), _exponent(largest))
            shift = tl.where((largest > 0) & (bound < float("inf")), tl.minimum(tl.maximum(shift, -120), 120), 0)
            tl.store(SHIFT, shift, mask=tl.program_id(0) == 0)
        else:
            shift = fixed_shift
        x = bf16_bits_to_fp32(tl.load(X + offsets, mask, 0)) * _power_of_two(shift)
        tl.store(Y + offsets, x.to(tl.float16), mask)

    @triton.jit(do_not_specialize=["weight_shift", "n", "channels", "spatial"])
    def _to_bf16_kernel(Y16, Y_BITS, BIAS, SHIFT, weight_shift, n, channels, spatial, HAS_BIAS: tl.constexpr, BLOCK: tl.constexpr):
        # FP16 result scaled back (two exact halves of the total power of two), bias added in FP32, rounded once to BF16 in place
        offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        total = -(tl.load(SHIFT) + weight_shift)
        half = total // 2
        y = tl.load(Y16 + offsets, mask, 0).to(tl.float32) * _power_of_two(half) * _power_of_two(total - half)
        if HAS_BIAS:
            y += bf16_bits_to_fp32(tl.load(BIAS + (offsets // spatial) % channels, mask, 0))
        tl.store(Y_BITS + offsets, fp32_to_bf16_bits(y), mask)

    # Register after definitions to preserve Triton's line-number-sensitive cache keys.
    from shared.kernels.triton_compilation_log import install_triton_compilation_logger
    install_triton_compilation_logger()
