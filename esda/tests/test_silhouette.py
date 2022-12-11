import unittest

import libpysal
import numpy
from sklearn.metrics import pairwise

from .. import silhouettes

RTOL = libpysal.common.RTOL
ATOL = libpysal.common.ATOL


class Silhouette_Tester(unittest.TestCase):
    def setUp(self):
        self.w = libpysal.weights.lat2W(3, 3, rook=False)
        numpy.random.seed(12345)
        self.X = numpy.random.random((9, 3))
        self.groups = numpy.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]]).flatten()
        self.precomputed = self.X @ self.X.T
        self.altmetric = pairwise.manhattan_distances

    def test_boundary(self):
        known = numpy.array(
            [
                0.03042804,
                -0.26071434,
                0.26664155,
                -0.09395997,
                0.02754655,
                -0.30581229,
                0.2697089,
                -0.29993007,
                0.0,
            ]
        )
        test = silhouettes.boundary_silhouette(self.X, self.groups, self.w)
        numpy.testing.assert_allclose(known, test, rtol=RTOL, atol=ATOL)
        known = numpy.array(
            [
                0.04246261,
                -0.17059996,
                0.35656939,
                -0.10765454,
                0.00831288,
                -0.30924291,
                0.36502035,
                -0.25464692,
                0.0,
            ]
        )
        test = silhouettes.boundary_silhouette(
            self.X, self.groups, self.w, metric=self.altmetric
        )
        numpy.testing.assert_allclose(known, test, rtol=RTOL, atol=ATOL)
        known = numpy.array(
            [
                -0.43402484,
                -0.29488836,
                -0.3663925,
                -0.32831205,
                -0.21536904,
                -0.10134236,
                -0.37144783,
                -0.01618446,
                0.0,
            ]
        )
        test = silhouettes.boundary_silhouette(
            self.X, self.groups, self.w, metric=self.precomputed
        )
        numpy.testing.assert_allclose(known, test, rtol=RTOL, atol=ATOL)
        with self.assertRaises(AssertionError):
            silhouettes.boundary_silhouette(
                self.X,
                self.groups,
                self.w,
                metric=self.precomputed - self.precomputed.mean(),
            )

    def test_path(self):
        known = numpy.array(
            [
                0.15982274,
                -0.02136909,
                -0.3972349,
                0.24479121,
                0.02754655,
                0.28465546,
                -0.07572727,
                0.26903733,
                0.4165144,
            ]
        )
        test = silhouettes.path_silhouette(self.X, self.groups, self.w)
        numpy.testing.assert_allclose(known, test, rtol=RTOL, atol=ATOL)
        known = numpy.array(
            [
                0.1520476,
                0.0390323,
                -0.34269345,
                0.27239358,
                0.00831288,
                0.27934432,
                -0.03874118,
                0.28623703,
                0.40062121,
            ]
        )
        test = silhouettes.path_silhouette(
            self.X, self.groups, self.w, metric=self.altmetric
        )
        numpy.testing.assert_allclose(known, test, rtol=RTOL, atol=ATOL)
        with self.assertRaises(TypeError):
            silhouettes.path_silhouette(
                self.X, self.groups, self.w, metric=self.precomputed
            )
        with self.assertRaises(AssertionError):
            silhouettes.path_silhouette(
                self.X, self.groups, self.w, metric=lambda d: -self.altmetric(d)
            )

    def test_nearest_label(self):
        known = numpy.array([1, 1, 1, 1, 0, 0, 1, 0, 0])
        test, d = silhouettes.nearest_label(self.X, self.groups, return_distance=True)
        numpy.testing.assert_array_equal(known, test)
        known = numpy.array([0, 1, 0, 0, 1, 0, 0, 0, 1])
        test = silhouettes.nearest_label(self.X, self.groups, keep_self=True)
        numpy.testing.assert_array_equal(test, known)
        knownd = numpy.array(
            [
                1.05707684,
                0.74780721,
                0.88841079,
                0.71628677,
                1.25964181,
                0.5854757,
                0.89710073,
                0.64575898,
                0.73913526,
            ]
        )
        numpy.testing.assert_allclose(d, knownd, rtol=RTOL, atol=ATOL)

    def test_silhouette_alist(self):
        known = numpy.array(
            [
                0.0,
                0.0,
                0.22434244,
                0.0,
                0.0,
                0.0,
                -0.07589293,
                -0.07589293,
                0.0,
                0.41331324,
                0.41331324,
                0.0,
                0.0,
                0.11703681,
                0.0,
                0.11703681,
                0.27065991,
                0.27065991,
                0.27065991,
                0.27065991,
                0.0,
                0.27065991,
                0.0,
                0.0,
                -0.07441639,
                -0.07441639,
                0.0,
                0.0,
                0.0,
                0.0,
                0.41576712,
                0.41576712,
                -0.06657343,
                0.0,
                0.0,
                -0.06657343,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )

        test = silhouettes.silhouette_alist(
            self.X, self.groups, self.w.to_adjlist(drop_islands=True)
        )
        numpy.testing.assert_allclose(known, test.silhouette, rtol=RTOL, atol=ATOL)
