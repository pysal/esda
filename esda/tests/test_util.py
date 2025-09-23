import libpysal
import numpy as np
import pytest

from .. import moran, util


class TestFdr:
    def setup_method(self):
        self.w = libpysal.io.open(libpysal.examples.get_path("stl.gal")).read()
        f = libpysal.io.open(libpysal.examples.get_path("stl_hom.txt"))
        self.y = np.array(f.by_col["HR8893"])

    def test_fdr(self):
        lm = moran.Moran_Local(
            self.y, self.w, transformation="r", permutations=999, seed=10
        )
        assert pytest.approx(util.fdr(lm.p_sim, 0.1)) == 0.002564102564102564
        assert pytest.approx(util.fdr(lm.p_sim, 0.05)) == 0.001282051282051282
