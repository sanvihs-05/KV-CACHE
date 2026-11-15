import numpy as np
import pytest

from kvwear.codecs import Codec, make_codec


def block(shape=(4, 2, 3, 16, 64), seed=0):
    return (np.random.default_rng(seed).standard_normal(shape) * 3).astype(np.float16)


def test_fp16_is_exact():
    x = block()
    assert np.array_equal(Codec.decode(make_codec("fp16").encode(x)), x)


@pytest.mark.parametrize("method,max_rel_err,max_ratio", [("int8", 0.01, 0.53), ("int4", 0.2, 0.29)])
@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_quantized_error_and_size(method, max_rel_err, max_ratio, backend):
    if backend == "torch":
        pytest.importorskip("torch")
    x = block()
    enc = Codec(method, backend=backend).encode(x)
    y = Codec.decode(bytearray(enc), backend=backend)
    assert y.shape == x.shape and y.dtype == np.float16
    g = x.astype(np.float32).reshape(-1, 64)           # error relative to each group's range
    rng = g.max(1) - g.min(1)
    err = np.abs(y.astype(np.float32).reshape(-1, 64) - g).max(1)
    assert (err <= max_rel_err * rng + 1e-3).all()
    assert len(enc) / x.nbytes < max_ratio


def test_backends_produce_identical_bytes():
    pytest.importorskip("torch")
    x = block(seed=3)
    for m in ("int8", "int4"):
        assert Codec(m, backend="torch").encode(x) == Codec(m, backend="numpy").encode(x)


def test_lossless_compression_roundtrip():
    x = block()
    for spec in ("fp16+zlib", "int4+zlib"):
        plain = Codec.decode(make_codec(spec.split("+")[0]).encode(x))
        assert np.array_equal(Codec.decode(make_codec(spec).encode(x)), plain)


def test_odd_head_dim_and_constant_groups():
    x = np.zeros((2, 2, 1, 3, 10), np.float16)       # head_dim 10 -> group falls back to 10
    x[..., 0] = 1
    for m in ("int8", "int4"):
        y = Codec.decode(Codec(m).encode(x))
        assert np.allclose(y, x, atol=0.07)


def test_rejects_garbage():
    with pytest.raises(ValueError):
        Codec.decode(b"nope" + b"\0" * 20)
