"""Teste modele de perete: energie, direcție, termalizare."""
import math

import numpy as np
import pytest

from brownian_sim.materials.wall_material import WallMaterial
from brownian_sim.physics.wall_models import ElasticBounce, DampedBounce, OUBounce, MaxwellDiffuse


class TestElasticBounce:
    def test_reverses_normal_preserves_tangent(self):
        m = ElasticBounce()
        v = np.array([1.0, 2.0, 0.0])
        n = np.array([1.0, 0.0, 0.0])
        vp = m.bounce_single(v, n, WallMaterial(e_n=1, beta_t=1), rng=None)
        assert vp[0] == pytest.approx(-1.0)
        assert vp[1] == pytest.approx(2.0)
        assert vp[2] == pytest.approx(0.0)

    def test_conserves_energy(self):
        m = ElasticBounce()
        rng = np.random.default_rng(7)
        for _ in range(50):
            v = rng.standard_normal(3)
            n = rng.standard_normal(3)
            n = n / np.linalg.norm(n)
            vp = m.bounce_single(v, n, WallMaterial(1, 1), rng=None)
            assert abs(np.dot(vp, vp) - np.dot(v, v)) < 1e-10


class TestDampedBounce:
    def test_dissipates_normal(self):
        m = DampedBounce()
        v = np.array([2.0, 0.0, 0.0])
        n = np.array([1.0, 0.0, 0.0])
        mat = WallMaterial(e_n=0.5, beta_t=1.0)
        vp = m.bounce_single(v, n, mat, rng=None)
        assert vp[0] == pytest.approx(-1.0)  # -e_n * 2 = -1
        assert vp[1] == pytest.approx(0.0)

    def test_dissipates_tangent(self):
        m = DampedBounce()
        v = np.array([0.0, 2.0, 0.0])
        n = np.array([1.0, 0.0, 0.0])
        mat = WallMaterial(e_n=1.0, beta_t=0.3)
        vp = m.bounce_single(v, n, mat, rng=None)
        assert vp[0] == pytest.approx(0.0)
        assert vp[1] == pytest.approx(0.6)  # 0.3 * 2


class TestOUBounce:
    def test_positive_normal_out(self):
        m = OUBounce(kT_over_m=1.0)
        rng = np.random.default_rng(42)
        mat = WallMaterial(e_n=0.5, beta_t=0.5)
        n = np.array([1.0, 0.0, 0.0])
        for _ in range(100):
            v = rng.standard_normal(3)
            # incidentă spre perete (v_n > 0 spre afară)
            v[0] = abs(v[0])
            vp = m.bounce_single(v, n, mat, rng)
            # după bounce, componenta pe n trebuie să fie pozitivă (spre interior)
            assert float(vp.dot(n)) > 0

    def test_thermalization_to_maxwell_boltzmann(self):
        """Aplicând OU bounce repetat cu incident gaussian, distribuția ieșirilor
        trebuie să fie MB la T țintă. Testez varianța per-componentă tangențială."""
        m = OUBounce(kT_over_m=1.0)
        rng = np.random.default_rng(42)
        mat = WallMaterial(e_n=0.5, beta_t=0.5)
        n = np.array([1.0, 0.0, 0.0])
        vs = []
        for _ in range(5000):
            v_in = rng.standard_normal(3)
            v_in[0] = abs(v_in[0])
            vs.append(m.bounce_single(v_in, n, mat, rng))
        V = np.array(vs)
        # tangențial (y, z): var ~ kT/m = 1
        assert abs(np.var(V[:, 1]) - 1.0) < 0.1
        assert abs(np.var(V[:, 2]) - 1.0) < 0.1

    def test_batch_matches_single(self):
        """Batch și single trebuie să dea rezultate cu aceeași statistică."""
        m = OUBounce(kT_over_m=1.0)
        rng = np.random.default_rng(42)

        N = 2000
        v_in = rng.standard_normal((N, 3))
        v_in[:, 0] = np.abs(v_in[:, 0])
        n = np.zeros((N, 3))
        n[:, 0] = 1.0
        e_n = np.full(N, 0.5)
        beta_t = np.full(N, 0.5)

        v_out = m.bounce_batch(v_in, n, e_n, beta_t, xp=np, rng=rng)
        # componenta pe n trebuie pozitivă
        assert (np.sum(v_out * n, axis=1) > 0).all()
        # varianță tangențială ~ 1
        assert abs(np.var(v_out[:, 1]) - 1.0) < 0.1
        assert abs(np.var(v_out[:, 2]) - 1.0) < 0.1


class TestMaxwellDiffuse:
    def test_forgets_input_velocity(self):
        """v_out trebuie să fie independent de v_in: corelația medie să fie ~0."""
        m = MaxwellDiffuse(kT_over_m=1.0)
        rng = np.random.default_rng(99)
        mat = WallMaterial(e_n=1.0, beta_t=1.0)
        n = np.array([1.0, 0.0, 0.0])
        v_out_list = []
        v_in_norms = []
        for _ in range(3000):
            v_in = rng.standard_normal(3)
            v_in[0] = abs(v_in[0])
            vp = m.bounce_single(v_in, n, mat, rng)
            v_out_list.append(vp)
            v_in_norms.append(np.linalg.norm(v_in))
        V = np.array(v_out_list)
        # corelația dintre |v_in| și componenta normală de ieșire trebuie să fie ~0
        corr = float(np.corrcoef(v_in_norms, V[:, 0])[0, 1])
        assert abs(corr) < 0.05

    def test_positive_normal_out(self):
        """Componenta normală de ieșire trebuie să fie strict pozitivă."""
        m = MaxwellDiffuse(kT_over_m=1.0)
        rng = np.random.default_rng(7)
        mat = WallMaterial(e_n=1.0, beta_t=1.0)
        n = np.array([1.0, 0.0, 0.0])
        for _ in range(200):
            v_in = rng.standard_normal(3)
            v_in[0] = abs(v_in[0])
            vp = m.bounce_single(v_in, n, mat, rng)
            assert float(vp.dot(n)) > 0

    def test_normal_rayleigh_distribution(self):
        """Componenta normală de ieșire urmează distribuția Rayleigh(s):
        media = s*sqrt(pi/2), varianța = s^2*(2 - pi/2), cu s = sqrt(kT/m)."""
        m = MaxwellDiffuse(kT_over_m=1.0)
        rng = np.random.default_rng(13)
        mat = WallMaterial(e_n=1.0, beta_t=1.0)
        n = np.array([1.0, 0.0, 0.0])
        vn_out = []
        for _ in range(5000):
            v_in = rng.standard_normal(3)
            v_in[0] = abs(v_in[0])
            vp = m.bounce_single(v_in, n, mat, rng)
            vn_out.append(float(vp.dot(n)))
        vn = np.array(vn_out)
        s = 1.0  # sqrt(kT/m) = 1
        expected_mean = s * math.sqrt(math.pi / 2)
        expected_var = s**2 * (2.0 - math.pi / 2.0)
        assert abs(np.mean(vn) - expected_mean) < 0.05
        assert abs(np.var(vn) - expected_var) < 0.05

    def test_tangential_gaussian(self):
        """Componenta tangențială de ieșire: varianță ~ kT/m = 1, independentă de v_in."""
        m = MaxwellDiffuse(kT_over_m=1.0)
        rng = np.random.default_rng(21)
        mat = WallMaterial(e_n=1.0, beta_t=1.0)
        n = np.array([1.0, 0.0, 0.0])
        v_out_list = []
        for _ in range(5000):
            v_in = rng.standard_normal(3)
            v_in[0] = abs(v_in[0])
            v_out_list.append(m.bounce_single(v_in, n, mat, rng))
        V = np.array(v_out_list)
        assert abs(np.var(V[:, 1]) - 1.0) < 0.1
        assert abs(np.var(V[:, 2]) - 1.0) < 0.1

    def test_batch_matches_single(self):
        """Implementarea batch produce aceeași statistică ca single-particle."""
        m = MaxwellDiffuse(kT_over_m=1.0)
        rng = np.random.default_rng(55)
        N = 3000
        v_in = rng.standard_normal((N, 3))
        v_in[:, 0] = np.abs(v_in[:, 0])
        n_arr = np.zeros((N, 3)); n_arr[:, 0] = 1.0
        e_n = np.ones(N, dtype=np.float32)
        beta_t = np.ones(N, dtype=np.float32)
        v_out = m.bounce_batch(v_in, n_arr, e_n, beta_t, xp=np, rng=rng)
        assert (np.sum(v_out * n_arr, axis=1) > 0).all()
        assert abs(np.var(v_out[:, 1]) - 1.0) < 0.1
        assert abs(np.var(v_out[:, 2]) - 1.0) < 0.1
