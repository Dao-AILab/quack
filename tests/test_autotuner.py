from quack.autotuner import AutotuneConfig


def test_autotune_config_supports_multi_kwarg_hash_and_equality():
    config_a = AutotuneConfig(block_m=128, num_warps=4)
    config_b = AutotuneConfig(block_m=128, num_warps=4)
    config_c = AutotuneConfig(block_m=64, num_warps=4)

    assert config_a == config_b
    assert hash(config_a) == hash(config_b)
    assert config_a != config_c

    timings = {config_a: 1.25, config_c: 2.5}
    assert timings[config_b] == 1.25
    assert len({config_a, config_b, config_c}) == 2
