# Python Extension for Creating Feature Vectors from LiDAR data using pgsql

This module contains hard-coded query processing in C++ for retrieving feature vectors (min, average, max) of
points given a context. Therefore,

* a range query is processed taking the 128x128m region around a query location into account.
* all points are sorted into 128x128 cells
* empty cells are filled using a kNN query (4 nearest neighbors)
* cell statistics is calculated (min, mean, max) and
* returned as a numpy array for deep learning

# Installation:
```
sudo python setup.py install
```
but before (if not on Linux) change the include path (-I) in setup.py to fit your postgres installation.
On Linux, you need libpostgres-dev package. You can find out some compiler settings with pg_config.


# Assumptions regarding the database

* This project uses a PostGIS database named *points* (changable via connection string from python).
* This database contains a table named points with columns (at least, more are always o.K.) X, Y, Z, geom, where
  geom is a geometry column for the spatial index. The data is retrieved using X, Y, Z such that transformations can go on before building the geom column, which is only used for the range query. For example, it would be o.K. to have
  a 2D geometry column for indexing or to have indexing in Euclidean space with WGS84 coordinates in X,Y,Z columns.

# Using it from python

Look at sample.py. You need to give a working connection string, two coordinates, the k for k-nearest-neighbour and the srid of your data and get back a numpy array
for the features.

# Sample Data:
You can decompress sample.dat to /tmp/sample.dat and import it:
```
COPY points(X,Y,Z) FROM '/tmp/sample.dat'
```
This should generate the same output as in this repository.


# Some sample of the database:

```
        x         |        y         |        z         |                                geom                                
------------------+------------------+------------------+--------------------------------------------------------------------
 545381.617274165 | 5800608.36359501 | 110.702255249023 | 01010000A0787F0000005C0B3CCBA4204100244517A8205641000000C0F1AC5B40
 545382.305749655 | 5800609.41563511 | 111.108291625977 | 01010000A0787F000000388B9CCCA4204100C4995AA820564100000040EEC65B40

```

Building the database:

```
CREATE TABLE points (X FLOAT, Y FLOAT, Z FLOAT);
COPY points(X,Y,Z) FROM '/home/martin/sample.dat' WITH DELIMITER ' ' CSV; 
SELECT AddGeometryColumn( 'points', 'geom', 32632, 'POINT', 3);
UPDATE points set geom = ST_SetSRID(ST_MakePoint(X, Y,Z),32632) WHERE geom is NULL;
CREATE INDEX points_geom ON points USING GIST(geom);
VACUUM ANALYZE
```

Note that you should use your projection, possibly reproject into some Euclidean space (as range queries are slow in
non-Euclidean projections). You are free to reproject only while setting geom and querying, you can still have
non-Euclidean data in the X,Y,Z columns.

Some sample queries (execute sample.py after compiling with defining DUMP_QUERIES)x

* Fast Range Query
```
SELECT x,y,z FROM points WHERE geom && ST_MakeEnvelope(545262.44579551497,5800477.7953092,545518.44579551497,5800733.7953092);
```
* kNN query (k=3)
```
SELECT * FROM points order by geom <#> ST_Point(545454.44579551497,5800669.7953092) LIMIT 3;
```
