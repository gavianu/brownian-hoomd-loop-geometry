"""Teste primitive geometrice: inside, wall_distance, snap_and_normal."""
import math

import numpy as np
import pytest

from brownian_sim.geometry.primitives import Box, CylX, CylY


class TestBox:
    def test_inside_center(self):
        b = Box(center=(0, 0, 0), size=(10, 10, 10))
        p = np.array([[0, 0, 0]], dtype=np.float64)
        assert b.inside(p)[0]

    def test_outside(self):
        b = Box(center=(0, 0, 0), size=(10, 10, 10))
        p = np.array([[10, 0, 0]], dtype=np.float64)
        assert not b.inside(p)[0]

    def test_on_boundary_inside(self):
        b = Box(center=(0, 0, 0), size=(10, 10, 10))
        p = np.array([[5.0, 0, 0]], dtype=np.float64)
        assert b.inside(p)[0]  # frontiera e inclusă (<=)

    def test_wall_distance(self):
        b = Box(center=(0, 0, 0), size=(10, 10, 10))
        # în centru, distanța = jumătate din cea mai mică dimensiune
        assert b.wall_distance(np.array([0.0, 0.0, 0.0])) == pytest.approx(5.0)
        # lângă fața x=5
        assert b.wall_distance(np.array([4.0, 0.0, 0.0])) == pytest.approx(1.0)

    def test_snap_and_normal_x_face(self):
        b = Box(center=(0, 0, 0), size=(10, 10, 10))
        p_new = np.array([6.0, 0.0, 0.0])  # depășește faza x+
        p_snap, n = b.snap_and_normal(p_new)
        assert p_snap[0] == pytest.approx(5.0)
        assert np.allclose(n, [1, 0, 0])

    def test_snap_and_normal_inside_returns_zero_normal(self):
        b = Box(center=(0, 0, 0), size=(10, 10, 10))
        p_new = np.array([0.0, 0.0, 0.0])
        p_snap, n = b.snap_and_normal(p_new)
        assert np.linalg.norm(n) < 1e-9


class TestCylX:
    def test_inside_axis(self):
        c = CylX(cx=0, cy=0, cz=0, R=5, L=10)
        assert c.inside(np.array([[0, 0, 0]]))[0]

    def test_outside_cap(self):
        c = CylX(cx=0, cy=0, cz=0, R=5, L=10)
        assert not c.inside(np.array([[6, 0, 0]]))[0]

    def test_outside_mantle(self):
        c = CylX(cx=0, cy=0, cz=0, R=5, L=10)
        assert not c.inside(np.array([[0, 6, 0]]))[0]

    def test_wall_distance_center(self):
        c = CylX(cx=0, cy=0, cz=0, R=5, L=10)
        # în centru: min(R, L/2) = 5
        assert c.wall_distance(np.array([0.0, 0.0, 0.0])) == pytest.approx(5.0)

    def test_snap_to_mantle(self):
        c = CylX(cx=0, cy=0, cz=0, R=5, L=10)
        p_new = np.array([0.0, 6.0, 0.0])
        p_snap, n = c.snap_and_normal(p_new)
        r = math.hypot(p_snap[1], p_snap[2])
        assert r == pytest.approx(5.0)
        assert n[0] == 0.0  # normala radială, fără componentă axială
        assert np.linalg.norm(n) == pytest.approx(1.0)

    def test_snap_to_cap(self):
        c = CylX(cx=0, cy=0, cz=0, R=5, L=10)
        p_new = np.array([6.0, 0.0, 0.0])
        p_snap, n = c.snap_and_normal(p_new)
        assert p_snap[0] == pytest.approx(5.0)
        assert np.allclose(n, [1, 0, 0])


class TestCylY:
    def test_inside(self):
        c = CylY(cx=0, cy=0, cz=0, R=5, L=10)
        assert c.inside(np.array([[0, 0, 0]]))[0]
        assert not c.inside(np.array([[0, 6, 0]]))[0]  # cap
        assert not c.inside(np.array([[6, 0, 0]]))[0]  # mantle


class TestAssembly:
    def test_sample_inside(self):
        from brownian_sim.geometry.presets.simple_box import build
        A = build()
        rng = np.random.default_rng(42)
        P = A.sample_uniform(500, rng)
        assert A.inside_any(P).all()

    def test_loop_chambers_all_inside(self):
        from brownian_sim.geometry.presets.loop_chambers import build
        A = build()
        rng = np.random.default_rng(42)
        P = A.sample_uniform(500, rng)
        assert A.inside_any(P).all()
