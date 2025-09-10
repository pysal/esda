import numpy as np

from .. import mixture_smoothing as m_s


class TestNPMixtureSmoother:
    """Mixture Smoothing Unit Tests"""

    def setup_method(self):
        self.e = np.array([10, 5, 12, 20])
        self.b = np.array([100, 150, 80, 200])

    def test_defaults(self):
        """Test the main class"""
        mix = m_s.NP_Mixture_Smoother(self.e, self.b)

        np.testing.assert_array_almost_equal(
            mix.r, np.array([0.10982278, 0.03445531, 0.11018404, 0.11018604])
        )
        np.testing.assert_array_almost_equal(mix.category, np.array([1, 0, 1, 1]))

        left, right = mix.getSeed()
        np.testing.assert_array_almost_equal(left, np.array([0.5, 0.5]))
        np.testing.assert_array_almost_equal(right, np.array([0.03333333, 0.15]))

        d = mix.mixalg()
        np.testing.assert_array_almost_equal(
            d["mix_den"], np.array([0.0, 0.0, 0.0, 0.0])
        )
        np.testing.assert_array_almost_equal(d["gradient"], np.array([0.0]))
        np.testing.assert_array_almost_equal(d["p"], np.array([1.0]))
        np.testing.assert_array_almost_equal(d["grid"], np.array([11.27659574]))

        assert d["k"] == 1
        assert d["accuracy"] == 1.0

        left, right = mix.getRateEstimates()
        np.testing.assert_array_almost_equal(
            left, np.array([0.0911574, 0.0911574, 0.0911574, 0.0911574])
        )
        np.testing.assert_array_almost_equal(right, np.array([1, 1, 1, 1]))
