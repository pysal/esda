import geopandas, libpysal, esda, numpy

df = geopandas.read_file(libpysal.examples.get_path("columbus.shp"))
y = df[["HOVAL"]].values
w = libpysal.weights.Queen.from_dataframe(df)
w.transform = 'r'
lmo = esda.moran.Moran_Local(y, w)
row, col = w.sparse.nonzero()
weight = w.sparse.data
# esda.moran.choice_neighbors(lmo.z, lmo.Is, row, weight, 1000, True)
# esda.moran.while_neighbors(lmo.z, lmo.Is, row, weight, 1000, True)
