import unittest
import pandas
import numpy as np
from .. import adbscan


class ADBSCAN_Tester(unittest.TestCase):
    def setUp(self):
        np.random.seed(10)
        self.db = pandas.DataFrame(
            {"x": np.random.random(25), "y": np.random.random(25)}
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
        # ------------------------#
        #           # Single Core #
        # ------------------------#
        np.random.seed(10)
        ads = adbscan.ADBSCAN(0.03, 3, reps=10, keep_solus=True)
        # test there's no labels_, votes, or solus
        _ = ads.fit(self.db, xy=["x", "y"])
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
        # ------------------------#
        #           # Multi Core #
        # ------------------------#
        np.random.seed(10)
        ads = adbscan.ADBSCAN(0.03, 3, reps=10, keep_solus=True, n_jobs=-1)
        _ = ads.fit(self.db, xy=["x", "y"])
        # Params
        self.assertEqual(ads.n_jobs, -1)
        # Labels
        np.testing.assert_equal(ads.labels_, self.lbls)
        # Votes
        votes = pandas.DataFrame({"lbls": self.lbls, "pct": self.pcts})
        np.testing.assert_equal(ads.votes["lbls"].values, self.lbls)
        np.testing.assert_almost_equal(ads.votes["pct"].values, votes["pct"].values)
        # Solus (only testing the sums as there're too many values)
        np.testing.assert_equal(ads.solus.astype(int).sum().sum(), 133)
        rep_sum = np.array([9, 24, 16, 13, 9, 16, 7, 13, 14, 12])
        np.testing.assert_equal(ads.solus.astype(int).sum().values, rep_sum)
        i_sum = np.array(
            [4, 8, 4, 5, 6, 6, 8, 5, 6, 6, 3, 2, 7, 6, 5, 4, 5, 6, 5, 4, 6, 6, 7, 3, 6]
        )
        np.testing.assert_equal(ads.solus.astype(int).sum(axis=1).values, i_sum)


class Remap_lbls_Tester(unittest.TestCase):
    def setUp(self):
        self.db = pandas.DataFrame({"X": [0, 0.1, 4, 6, 5], \
                               "Y": [0, 0.2, 5, 7, 5] \
                              })
        self.solus = pandas.DataFrame({"rep-00": [0, 0, 7, 7, -1], \
                                  "rep-01": [4, 4, -1, 6, 6], \
                                  "rep-02": [5, 5, 8, 8, 8] \
                                 })
    def test_remap_lbls(self):
        vals = np.array([[0, 0, 0], [0, 0, 0], [7, -1, 7], [7, 7, 7], [-1, 7, 7]])
        # ------------------------#
        #           # Single Core #
        # ------------------------#
        lbls = adbscan.remap_lbls(self.solus, self.db)
        # Column names
        np.testing.assert_equal(self.solus.columns.values, lbls.columns.values)
        # Values
        np.testing.assert_equal(lbls.values, vals)
        # ------------------------#
        #            # Multi Core #
        # ------------------------#
        lbls = adbscan.remap_lbls(self.solus, self.db, n_jobs=-1)
        # Column names
        np.testing.assert_equal(self.solus.columns.values, lbls.columns.values)
        # Values
        np.testing.assert_equal(lbls.values, vals)
 

class Ensemble_Tester(unittest.TestCase):
    def setUp(self):
        self.db = pandas.DataFrame({"X": [0, 0.1, 4, 6, 5], \
                               "Y": [0, 0.2, 5, 7, 5] \
                              })\
                              .rename(lambda i: 'i_'+str(i))
        self.solus = pandas.DataFrame({"rep-00": [0, 0, 7, 7, -1], \
                                  "rep-01": [4, 4, -1, 6, 6], \
                                  "rep-02": [5, 5, 8, 8, 8] \
                                 })\
                              .rename(lambda i: 'i_'+str(i))
    def test_ensemble(self):
        vals = np.array([[0.0, 1.0], [0.0, 1.0], [7.0, 0.6666666666666666], [7.0, 1.0], [7.0, 0.6666666666666666]])
        # ------------------------#
        #           # Single Core #
        # ------------------------#
        ensemble_solu = adbscan.ensemble(self.solus, self.db)
        # Column names
        np.testing.assert_equal(ensemble_solu.columns.values.tolist(), ['lbls', 'pct'])
        # Index
        # Values
        np.testing.assert_almost_equal(ensemble_solu.values, vals)
        # ------------------------#
        #           # Multi  Core #
        # ------------------------#
        ensemble_solu = adbscan.ensemble(self.solus, self.db, n_jobs=-1)
        # Column names
        np.testing.assert_equal(ensemble_solu.columns.values.tolist(), ['lbls', 'pct'])
        # Values
        np.testing.assert_almost_equal(ensemble_solu.values, vals)
  

suite = unittest.TestSuite()
test_classes = [ADBSCAN_Tester, Remap_lbls_Tester, Ensemble_Tester]
for i in test_classes:
    a = unittest.TestLoader().loadTestsFromTestCase(i)
    suite.addTest(a)

if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite)
