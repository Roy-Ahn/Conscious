import numpy as np

from conscious.cortex import Cortex


def batch(rng, vocab, window, size=5):
    ctx = rng.integers(0, vocab, size=(size, window)).astype(np.int32)
    ctx[0, 3:] = -1
    return ctx, rng.integers(0, vocab, size=size)


def test_gradients_match_finite_differences():
    rng = np.random.default_rng(0)
    cortex = Cortex(vocab_size=7, dim=4, hidden=5, window=4, seed=1)
    cortex.params["W2"] = rng.normal(0, 0.5, cortex.params["W2"].shape)
    ctx, targets = batch(rng, 7, 4)
    _, _, grads = cortex.gradients(ctx, targets)
    eps = 1e-5
    for name, param in cortex.params.items():
        for _ in range(6):
            index = tuple(rng.integers(0, n) for n in param.shape)
            keep = param[index]
            param[index] = keep + eps
            up = cortex.gradients(ctx, targets)[0]
            param[index] = keep - eps
            down = cortex.gradients(ctx, targets)[0]
            param[index] = keep
            assert abs((up - down) / (2 * eps) - grads[name][index]) < 1e-6, name


def test_learning_lowers_the_loss():
    rng = np.random.default_rng(0)
    cortex = Cortex(vocab_size=10, dim=8, hidden=16, window=3, lr=0.02)
    ctx, targets = batch(rng, 10, 3, size=8)
    first = cortex.learn(ctx, targets)[0]
    for _ in range(200):
        last = cortex.learn(ctx, targets)[0]
    assert last < 0.1 * first


def test_growing_adds_units_without_changing_predictions():
    rng = np.random.default_rng(0)
    cortex = Cortex(vocab_size=10, dim=8, hidden=6, window=3)
    ctx, targets = batch(rng, 10, 3)
    for _ in range(20):
        cortex.learn(ctx, targets)
    before, size = cortex.predict(ctx[1]).copy(), cortex.n_params
    cortex.grow(4)
    assert cortex.hidden == 10
    assert cortex.n_params > size
    np.testing.assert_allclose(cortex.predict(ctx[1]), before)
    cortex.learn(ctx, targets)
    assert np.abs(cortex.params["W2"][:, 6:]).max() > 0


def test_new_words_add_rows_and_keep_old_preferences():
    rng = np.random.default_rng(0)
    cortex = Cortex(vocab_size=6, dim=8, hidden=6, window=3)
    ctx, targets = batch(rng, 6, 3)
    for _ in range(20):
        cortex.learn(ctx, targets)
    before = cortex.predict(ctx[1]).copy()
    assert cortex.ensure_vocab(9) == 3
    after = cortex.predict(ctx[1])
    assert after.shape == (9,)
    np.testing.assert_allclose(after[:6] / after[:6].sum(), before / before.sum())


def test_pruning_removes_only_units_that_carry_nothing():
    rng = np.random.default_rng(0)
    cortex = Cortex(vocab_size=10, dim=8, hidden=12, window=3)
    ctx, targets = batch(rng, 10, 3)
    for _ in range(20):
        cortex.learn(ctx, targets)
    cortex.params["W2"][:, [2, 7, 9]] = 0.0
    before = cortex.predict(ctx[1]).copy()
    assert cortex.prune() == 3
    assert cortex.hidden == 9
    np.testing.assert_allclose(cortex.predict(ctx[1]), before)
    cortex.learn(ctx, targets)
