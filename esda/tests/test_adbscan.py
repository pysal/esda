import unittest
import pandas
import numpy as np
from .. import adbscan


class ADBSCAN_Tester(unittest.TestCase):
    def setUp(self):
        np.random.seed(10)
        self.db = pandas.DataFrame(
            {"X": np.random.random(25), "Y": np.random.random(25)}
        )
        self.lbls = np.array(
            [
                "-1",
                "-1",
                "-1",
                "0",
                "-1",
                "-1",
                "-1",
                "0",
                "-1",
                "-1",
                "-1",
                "-1",
                "-1",
                "-1",
                "0",
                "0",
                "0",
                "-1",
                "0",
                "-1",
                "0",
                "-1",
                "-1",
                "-1",
                "-1",
            ],
            dtype=object,
        )
        self.pcts = np.array(
            [
                0.7,
                0.5,
                0.7,
                1.0,
                0.7,
                0.7,
                0.5,
                1.0,
                0.7,
                0.7,
                0.6,
                0.6,
                0.6,
                0.7,
                1.0,
                0.9,
                1.0,
                0.7,
                1.0,
                0.7,
                0.9,
                0.7,
                0.8,
                0.6,
                0.7,
            ]
        )

    def test_adbscan(self):
        np.random.seed(10)
        ads = adbscan.ADBSCAN(0.03, 3, reps=10, keep_solus=True)
        _ = ads.fit(self.db)
        # Params
        self.assertAlmostEqual(ads.eps, 0.03)
        self.assertEqual(ads.min_samples, 3)
        self.assertEqual(ads.algorithm, "auto")
        self.assertEqual(ads.n_jobs, 1)
        self.assertEqual(ads.pct_exact, 0.1)
        self.assertEqual(ads.reps, 10)
        self.assertEqual(ads.keep_solus, True)
        self.assertEqual(ads.pct_thr, 0.9)
        # Labels
        np.testing.assert_equal(ads.labels_, self.lbls)
        # Votes
        votes = pandas.DataFrame({"lbls": self.lbls, "pct": self.pcts})
        np.testing.assert_equal(ads.votes["lbls"].values, self.lbls)
        np.testing.assert_almost_equal(ads.votes["pct"].values, votes["pct"].values)
        # Solus
        np.testing.assert_equal(ads.solus.astype(int).sum().sum(), 133)
        rep_sum = np.array([9, 24, 16, 13, 9, 16, 7, 13, 14, 12])
        np.testing.assert_equal(ads.solus.astype(int).sum().values, rep_sum)
        i_sum = np.array(
            [4, 8, 4, 5, 6, 6, 8, 5, 6, 6, 3, 2, 7, 6, 5, 4, 5, 6, 5, 4, 6, 6, 7, 3, 6]
        )
        np.testing.assert_equal(ads.solus.astype(int).sum(axis=1).values, i_sum)


class Remap_lbls_Tester(unittest.TestCase):
    pass


class Ensemble_Tester(unittest.TestCase):
    pass


suite = unittest.TestSuite()
test_classes = [ADBSCAN_Tester, Remap_lbls_Tester, Ensemble_Tester]
for i in test_classes:
    a = unittest.TestLoader().loadTestsFromTestCase(i)
    suite.addTest(a)

if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite)
