from unittest import TestCase

import numpy
import pandas

from ..topo import isolation, prominence, to_elevation, weights


class TopoTester(TestCase):
    def setUp(self):
        self.points = numpy.array(
            [[0, 0], [0, 1], [1, 1], [2, 0.5], [0.5, 0.5], [0.75, 0]]
        )
        self.marks = numpy.array([-1, 0.5, 1, 2, 3, 1.25])
        self.cxn = weights.Voronoi(self.points)

    def test_prominence_valid(self):
        w = self.cxn
        marks = self.marks

        prom = prominence(marks, w, verbose=False, progressbar=False)

        assert numpy.isnan(prom).sum().item() == 3
        assert (prom == 0).sum().item() == 1
        assert prom[-2] == 1.75
        assert prom[-3] == 0.75

        marks2 = marks.copy()
        marks2[-2] = -3

        prom = prominence(marks2, w, verbose=False, progressbar=False)

        assert prom[~numpy.isnan(prom)].max() == 5
        assert prom[~numpy.isnan(prom)].min() == 0
        assert numpy.isnan(prom).sum() == 3

    def test_prominence_options(self):
        marks = self.marks
        cxn = self.cxn

        default = prominence(marks, cxn)
        retvals = prominence(marks, cxn, return_all=True)
        middle = prominence(marks, cxn, middle="median")

        assert isinstance(default, numpy.ndarray)
        assert isinstance(retvals, pandas.DataFrame)
        numpy.testing.assert_array_equal(default, retvals.prominence.values)
        assert not numpy.allclose(default, middle)

    def test_isolation_options(self):
        marks = self.marks
        points = self.points
        default = isolation(marks, points)
        retvals = isolation(marks, points, return_all=True)
        metrics = isolation(marks, points, metric="haversine")
        middle = isolation(marks, points, middle="median")

        assert isinstance(default, numpy.ndarray)
        assert isinstance(retvals, pandas.DataFrame)
        numpy.testing.assert_array_equal(default, retvals.isolation)
        assert not numpy.allclose(default, metrics)
        assert not numpy.allclose(default, middle)

    def test_isolation_valid(self):
        # results should be valid

        marks = self.marks
        points = self.points

        iso = isolation(marks, points, return_all=True).assign(marks=marks)

        assert iso.loc[0, "index"] == 0
        assert numpy.isnan(iso.loc[4, "parent_rank"])
        assert (iso.dropna().parent_index == 4).all()
        assert (
            iso.sort_values("marks", ascending=False).index == (
                iso.sort_values("rank").index
            )
        ).all()
        assert iso.loc[3, "isolation"] == 1.5
        assert iso.loc[2, "gap"] == (
            marks[iso.loc[2, "parent_index"].astype(int)] - marks[2]
        )

        marks2 = self.marks.copy()
        marks2[-2] = 0

        iso = isolation(marks2, points, return_all=True).assign(marks=marks2)

        assert iso.loc[0, "index"] == 0
        assert numpy.isnan(iso.loc[3, "parent_index"])
        assert (iso.dropna().parent_index == [4, 2, 5, 5, 3]).all()
        assert (
            iso.sort_values("marks", ascending=False).index == (
                iso.sort_values("rank").index
            )
        ).all()
        assert iso.loc[1, "isolation"] == 1
        assert iso.loc[2, "gap"] == (
            marks2[iso.loc[2, "parent_index"].astype(int)] - marks2[2]
        )

    def test_to_elevation(self):
        onedim = to_elevation(self.marks)
        twodim = to_elevation(self.points)
        random = to_elevation(numpy.random.normal(size=(100, 4)))

        assert onedim.ndim == 1
        assert onedim.min() == 0
        assert onedim.max() == 4
        shift = self.marks - self.marks.min()
        assert (onedim == shift).all()

        assert twodim.ndim == 1
        assert twodim.min() == 0
        assert twodim.max() > 1

        assert random.ndim == 1
        assert random.min() >= 0
        assert random.max() >= 0
