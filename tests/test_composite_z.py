import jax
import jax.numpy as jnp
import numpy as onp

from chalk.style import composite, composite_by_z, soft_perm_ascending, to_color


def test_soft_perm_recovers_argsort():
    z = jnp.array([0.4, -1.2, 2.0, 0.1])
    p = soft_perm_ascending(z, temperature=0.02)
    hard = jnp.argsort(z)
    pred = jnp.argmax(p, axis=-1)
    onp.testing.assert_array_equal(onp.asarray(pred), onp.asarray(hard))
    onp.testing.assert_allclose(onp.asarray(p.sum(axis=-1)), 1.0, atol=1e-5)


def test_composite_by_z_hard_matches_scan_over():
    white = jnp.ones((4, 4, 3))
    cov = jnp.zeros((2, 4, 4))
    cov = cov.at[0, :, :].set(1.0)
    cov = cov.at[1, 1:3, 1:3].set(1.0)
    rgb = jnp.stack([to_color("green"), to_color("red")])
    opac = jnp.array([0.5, 0.9])
    z = jnp.array([0.0, 1.0])
    out = composite_by_z(white, cov, (rgb, opac), z, hard=True)
    ref = composite(white, cov[0], (rgb[0], opac[0]))
    ref = composite(ref, cov[1], (rgb[1], opac[1]))
    onp.testing.assert_allclose(onp.asarray(out), onp.asarray(ref), atol=1e-6)


def test_composite_by_z_soft_near_hard():
    white = jnp.ones((6, 6, 3))
    cov = jnp.zeros((2, 6, 6))
    cov = cov.at[0].set(0.8)
    cov = cov.at[1, 2:5, 2:5].set(1.0)
    rgb = jnp.stack([to_color("green"), to_color("red")])
    opac = jnp.array([0.55, 0.95])
    z = jnp.array([-2.0, 2.0])
    soft = composite_by_z(white, cov, (rgb, opac), z, temperature=0.05)
    hard = composite_by_z(white, cov, (rgb, opac), z, hard=True)
    onp.testing.assert_allclose(onp.asarray(soft), onp.asarray(hard), atol=0.05)


def test_composite_by_z_grad_z():
    white = jnp.ones((8, 8, 3))
    cov = jnp.zeros((2, 8, 8))
    cov = cov.at[0, :, :].set(1.0)
    cov = cov.at[1, 2:6, 2:6].set(1.0)
    rgb = jnp.stack([to_color("green"), jnp.array([0.4, 0.15, 0.05])])
    opac = jnp.array([0.6, 0.95])
    goal = jnp.zeros((8, 8, 3))

    def loss(z):
        img = composite_by_z(white, cov, (rgb, opac), z, temperature=0.35)
        return jnp.sum((img - goal) ** 2)

    z = jnp.array([-0.2, 0.2])
    g = jax.jit(jax.grad(loss))(z)
    assert g.shape == (2,)
    assert bool(jnp.isfinite(g).all())
    assert float(jnp.linalg.norm(g)) > 1e-6
