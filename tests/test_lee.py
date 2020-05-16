import unittest
import libpysal
import geopandas
from libpysal.common import pandas, RTOL, ATOL
from .. import lee
import numpy


PANDAS_EXTINCT = pandas is None

class Lee_Tester(unittest.TestCase):
    def setUp(self):
        self.data = geopandas.read_file(libpysal.examples.get_path("columbus.shp"))
        self.w = libpysal.weights.Queen.from_dataframe(self.data)
        self.w.transform = 'r'
        self.x = self.data[['HOVAL']].values
        self.y = self.data[['CRIME']].values

    def test_global(self):
        numpy.random.seed(2478879)
        result = lee.Spatial_Pearson(connectivity=self.w.sparse).fit(self.x, self.y)
        known = numpy.array([[ 0.30136527, -0.23625603],
                             [-0.23625603,  0.53512008]])
        numpy.testing.assert_allclose(known,result.association_, rtol=RTOL, atol=ATOL)
        numpy.testing.assert_array_equal(result.reference_distribution_.shape, (999,2,2))
        first_rep = numpy.array([[ 0.22803705, -0.08053692],
                                 [-0.08053692,  0.18897318]])

        second_rep = numpy.array([[ 0.14179274, -0.06962692],
                                  [-0.06962692,  0.13688337]])
        numpy.testing.assert_allclose(first_rep, result.reference_distribution_[0],
                                      rtol=RTOL, atol=ATOL)
        numpy.testing.assert_allclose(second_rep, result.reference_distribution_[1],
                                      rtol=RTOL, atol=ATOL)

        known_significance = numpy.array([[0.125, 0.026],
                                          [0.026, 0.001]])
        numpy.testing.assert_allclose(known_significance, result.significance_, 
                                      rtol=RTOL, atol=ATOL)

    def test_local(self):
        numpy.random.seed(2478879)
        result = lee.Spatial_Pearson_Local(connectivity=self.w.sparse).fit(self.x, self.y)
        known_locals = numpy.array([ 0.10246023, -0.24169198, -0.1308714 ,  
                                     0.00895543, -0.16080899, -0.00950808, 
                                     -0.14615398, -0.0627634 ,  0.00661232, 
                                     -0.42354628, -0.73121006,  0.02060548,  
                                     0.05187356,  0.06515283, -0.64400723,
                                    -0.37489818, -2.06573667, -0.10931854,  
                                    0.50823848, -0.06338637, -0.10559429,  
                                    0.03282849, -0.86618915, -0.62333825, 
                                    -0.40910044,-0.41866868, -0.00702983, 
                                    -0.4246288 , -0.52142507, -0.22481772,
                                    0.1931263 , -1.39355214,  0.02036755,  
                                    0.22896308, -0.00240854, -0.30405211, 
                                    -0.66950406, -0.21481868, -0.60320158, 
                                    -0.38117303, -0.45584563,  0.32019362, 
                                    -0.02818729, -0.02214172,  0.05587915,
                                    0.0295999 , -0.78818135,  0.16854472,  
                                    0.2378127 ])
        numpy.testing.assert_allclose(known_locals, result.associations_, 
                                      rtol=RTOL, atol=ATOL)
        significances = numpy.array([0.154, 0.291, 0.358, 0.231, 0.146, 
                                     0.335, 0.325, 0.388, 0.244, 0.111, 
                                     0.019, 0.165, 0.136, 0.073, 0.014, 
                                     0.029, 0.002, 0.376, 0.003, 0.265, 
                                     0.449, 0.121, 0.072, 0.006, 0.036, 
                                     0.06 , 0.355, 0.01 , 0.017, 0.168, 
                                     0.022, 0.003, 0.217, 0.016, 0.337, 
                                     0.137, 0.015, 0.128, 0.11 , 0.09 , 
                                     0.168, 0.031, 0.457, 0.44 , 0.141,
                                     0.249, 0.158, 0.018, 0.031])
        numpy.testing.assert_allclose(significances, result.significance_,
                                      rtol=RTOL, atol=ATOL)

suite = unittest.TestSuite()
test_classes = [Lee_Tester] 
for i in test_classes:
    a = unittest.TestLoader().loadTestsFromTestCase(i)
    suite.addTest(a)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
