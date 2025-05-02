import numpy
import esda
import pandas
from libpysal.weights import Voronoi

#this was more a validation exercise than a test. 
#Set tests to check:
#1. the results of the two sided are always greater than the directed
#2. the results of the directed are equal to either the lesser or greater
#3. the folded variant is close to the two-sided variant in a normal problem, 
#   but is similar to the one-sided test in a very skewed problem
#4. all p-values are between 0 and 1, with some p-values near 1
#5. no directed p-value will be bigger than .5

def test_significance():
    raise NotImplementedError()

"""
coordinates = numpy.random.random(size=(2000, 2))
x = numpy.random.normal(size=(2000,))
w = Voronoi(coordinates, clip="bbox")
w.transform = "r"
stat = esda.Moran_Local(x, w)

ts = calculate_significance(stat.Is, stat.rlisas, method="two-sided")
di = calculate_significance(stat.Is, stat.rlisas, method="directed")
lt = calculate_significance(stat.Is, stat.rlisas, method="lesser")
gt = calculate_significance(stat.Is, stat.rlisas, method="greater")
fo = calculate_significance(stat.Is, stat.rlisas, method="folded")

numpy.testing.assert_array_equal(
    numpy.minimum(lt, gt), di
)  # di is just the minimum of the two tests

print(
    f"directed * 2 is the same as two-sided {(di*2 == ts).mean()*100}% of the time"
)

print(
    pandas.DataFrame(
        numpy.column_stack((ts, di, fo, lt, gt)),
        columns=["two-sided", "directed", "folded", "lt", "gt"],
    ).corr()
)

answer = input("run big simulation? [y/n]")
if answer.lower().startswith("y"):
    all_correlations = []
    for i in range(1000):
        x = numpy.random.normal(size=(2000,))
        stat = esda.Moran_Local(x, w)
        ts = calculate_significance(stat.Is, stat.rlisas, method="two-sided")
        di = calculate_significance(stat.Is, stat.rlisas, method="directed")
        lt = calculate_significance(stat.Is, stat.rlisas, method="lesser")
        gt = calculate_significance(stat.Is, stat.rlisas, method="greater")
        fo = calculate_significance(stat.Is, stat.rlisas, method="folded")
        corrs = (
            pandas.DataFrame(
                numpy.column_stack((ts, di, fo, lt, gt)),
                columns=["two-sided", "directed", "folded", "lt", "gt"],
            )
            .corr()
            .assign(repno=i)
        )
        all_correlations.append(corrs)
    all_correlations = pandas.concat(all_correlations)
"""
