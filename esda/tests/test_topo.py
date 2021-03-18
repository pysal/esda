import numpy, copy, pandas
from unittest import TestCase

from ..topo import prominence, isolation, to_elevation, weights


class TopoTester(TestCase):
    def setUp():
        self.points = numpy.array(
            [[0, 0], [0, 1], [1, 1], [2, 0.5], [0.5, 0.5], [0.75, 0]]
        )
        self.marks = numpy.array([-1, 0.5, 1, 2, 3, 1.25])
        self.cxn = weights.Voronoi(self.points)

    def test_prominence_valid():
        # result should be valid
        ...

    def test_prominence_options():
        # return options should return correctly
        ...

    def test_isolation_options():
        default = isolation(marks, points)
        retvals = isolation(marks, points, return_all=True)
        metrics = isolation(marks, points, metric="haversine")
        middle = isolation(marks, points, middle="median")

        assert isinstance(default, numpy.ndarray)
        assert isinstance(retvals, pandas.DataFrame)
        assert not numpy.allclose(default, metrics)
        assert not numpy.allclose(default, middle)

    def test_isolation_valid():
        # results should be valid

        marks = self.marks
        points = self.points

        iso = isolation(marks, points, return_all=True).assign(marks=marks)

        assert iso.loc[0, "index"] == 0
        assert numpy.isnan(iso.loc[4, "parent_rank"])
        assert (iso.dropna().parent_index == 4).all()
        assert (
            iso.sort_values("marks", ascending=False).index
            == iso.sort_values("rank").index
        ).all()
        assert iso.loc[3, "distance"] == 1.5
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
            iso.sort_values("marks", ascending=False).index
            == iso.sort_values("rank").index
        ).all()
        assert iso.loc[1, "distance"] == 1
        assert iso.loc[2, "gap"] == (
            marks2[iso.loc[2, "parent_index"].astype(int)] - marks2[2]
        )

    def test_to_elevation():
        # middle change should work
        ...
