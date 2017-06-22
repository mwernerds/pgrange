/*
(c) 2017 M. Werner


*/

#define BENCHMARKING
//#define DUMP_QUERIES

#include<Python.h>
#include<iostream>
#include<sstream>
#include<limits>
#include<vector>
#include<chrono>
#include<numeric>
#include <libpq-fe.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

using namespace std;

static void
exit_nicely(PGconn *conn)
{
	PQfinish(conn);
	exit(1);
}

/*
Two query templates with full floating point precision...
*/
std::string range_query(double xmin, double ymin, double xmax, double ymax, int srid)
{
	std::stringstream ss;
	ss.precision(std::numeric_limits<double>::max_digits10);

	ss << "SELECT X,Y,Z FROM points WHERE ST_within(geom, ST_SetSRID(ST_MakeBox2D(ST_Point(" << xmin << "," << ymin << "), ST_Point(" << xmax << "," << ymax << "))," << srid << "));";
	return(ss.str());
}

std::string knn_query(double x, double y, int k, int srid)
{
	std::stringstream ss;
	ss.precision(std::numeric_limits<double>::max_digits10);
	ss << "SELECT * FROM points order by geom <-> ST_SetSRID(ST_Point(" << x << "," << y << ")," << srid << ") LIMIT " << k << ";";
	return(ss.str());
}

/*Binary Protocol Point Retrieval with helper functions (endianness for double, etc.)*/
double double_swap(double d)
{
	union
	{
		double d;
		unsigned char bytes[8];
	} src, dest;

	src.d = d;
	dest.bytes[0] = src.bytes[7];
	dest.bytes[1] = src.bytes[6];
	dest.bytes[2] = src.bytes[5];
	dest.bytes[3] = src.bytes[4];
	dest.bytes[4] = src.bytes[3];
	dest.bytes[5] = src.bytes[2];
	dest.bytes[6] = src.bytes[1];
	dest.bytes[7] = src.bytes[0];
	return dest.d;
}

double fieldValue(PGresult *res, int i, int XIndex)
{

	char *bptr = PQgetvalue(res, i, XIndex);
	double x = double_swap(*((double*)bptr));
	return x;
}

/*
Converts SQL result into array object
Order: point by point, x,y,z for each point
*/
std::vector<double> pointsFromSQL(PGconn *conn, std::string query)
{

#ifdef DUMP_QUERIES
	cout << query << endl;
#endif
	PGresult   *res;
	res = PQexecParams(conn,
		query.c_str(),
		//query,
		0,       /* one param */
		NULL,    /* let the backend deduce param type */
		NULL,//paramValues,
		NULL,    /* don't need param lengths since text */
		NULL,    /* default to all text params */
		1);      /* ask for binary results */

	if (PQresultStatus(res) != PGRES_TUPLES_OK)
	{
		fprintf(stderr, "SELECT failed: %s", PQerrorMessage(conn));
		PQclear(res);
		exit_nicely(conn);
	}


	int N = PQntuples(res);
	std::vector<double> ret(3 * N);
	int idx_x = PQfnumber(res, "X");
	int idx_y = PQfnumber(res, "Y");
	int idx_z = PQfnumber(res, "Z");

	for (int i = 0; i < N; i++)
	{
		ret[i * 3] = fieldValue(res, i, idx_x);
		ret[i * 3 + 1] = fieldValue(res, i, idx_y);
		ret[i * 3 + 2] = fieldValue(res, i, idx_z);

	}


	PQclear(res);
	return ret;
}

/* Now your specific example*/

#include<map>
#include<algorithm>

/*Range query in order to find environment of a point. Default: +-64 meter in x- and y- direction around the point*/
std::string area_query(double x, double y, double radius, int srid)
{
	return range_query(x - radius, y - radius, x + radius, y + radius, srid);
}

/*
calculates minimal, average, maximal heights of a given origin point environment, defined by +-radius [m] in x- and y- direction.
Finds k nearest neighbour points if the environment is not homogeneous
*/
std::vector<double> calculateFeatures(PGconn *conn, double xorigin = 33313000.000, double yorigin = 5992000.000, double radius = 64.0, int k = 4, int srid = 5650)
{
	//return vector
	std::vector<double> ret;

	//get points within a certain region
	auto region = pointsFromSQL(conn, area_query(xorigin, yorigin, radius, srid));

	/* now, we need a function to calculate the cell coordinates for each point.
	Cell coordinates are located at the left bottom edge of a cell. cell<0,0> at coordinates equals Point(xorigin - radius, yorigin - radius)
	*/
	auto homogenous = [xorigin, yorigin, radius](double x, double y) {return std::make_pair((int)(x - (xorigin - radius)), (int)(y - (yorigin - radius))); };
	auto dehomogenous = [xorigin, yorigin, radius](std::pair<int, int> p) {return std::make_pair((p.first + 0.5 + (xorigin - radius)), (p.second + 0.5 + (yorigin - radius))); };

	//Maps all points of the region into cells, where <0,0> describes the left bottom edge of the region
	std::map<std::pair<int, int>, std::vector<int>>  cells;
	for (size_t i = 0; i < region.size() / 3; i++)
	{
		auto hc = homogenous(region[i * 3], region[i * 3 + 1]);
		cells[hc].push_back(i);
	}

	//function to get the heights out of the indices within "region" variable
	auto i2z = [&region](size_t i) {return region[i * 3 + 2]; };

	//calculate the minimal, mean and maximal height within each cell
	std::vector<double> Zvalues;
	for (int iy = 2 * radius - 1; iy >= 0; iy--)	//f.e radius = 64: 127->0 = upper boundary to bottom boundary of the region
		for (int ix = 0; ix < 2 * radius; ix++)		//f.e. radius = 64: 0->127 = left boundary to right boundary of the region
		{
			Zvalues.clear(); // just for debugging.
			auto hc = std::make_pair(ix, iy); // homogenous coordinates of cell
			auto a = cells[hc];

			/*
			when there are no points within the cell: Call knn-query and find the "k" nearest points
			Gets a pair of coordinates, which are half a meter righter and upper than the coordinates of <idx,idy>
			!!!Assuming: cell length equals one meter in the coordinate system!!!
			*/
			if (a.empty())
			{

				auto cp = dehomogenous(hc);
				auto q = knn_query(cp.first, cp.second, k, srid);
				auto knn = pointsFromSQL(conn, q);
				Zvalues.resize(knn.size() / 3);
				for (size_t i = 0; i < knn.size() / 3; i++)
					Zvalues[i] = knn[i * 3 + 2];
			}
			else {
				Zvalues.resize(a.size());
				std::transform(a.begin(), a.end(), Zvalues.begin(), i2z);
			}
			// now statistics on Zvalues
			double average = std::accumulate(std::begin(Zvalues), std::end(Zvalues), 0.0) / std::distance(std::begin(Zvalues), std::end(Zvalues));
			auto minmax = std::minmax_element(std::begin(Zvalues), std::end(Zvalues));
			
			//push minimal, average, maximal values of the cell into the return vector ret
			ret.push_back(*(minmax.first));
			ret.push_back(average);
			ret.push_back(*(minmax.second));

		}

	return ret;
}

// Load a dataset from a Shape File.
//Args: double xorigin, double yorigin, double radius, int k, int srid
static PyObject * pgrange_performQuery(PyObject *self, PyObject *args)
{
	double xorigin, yorigin, radius;
	int k, srid;
	std::vector<double> featureVector;
	const char *conninfo;
	if (!PyArg_ParseTuple(args, "sdddii", &conninfo, &xorigin, &yorigin, &radius, &k, &srid))
		return NULL;

	PGconn *conn;
	/* Make a connection to the database */
	conn = PQconnectdb(conninfo);
	/* Check to see that the backend connection was successfully made */
	if (PQstatus(conn) != CONNECTION_OK)
	{
		fprintf(stderr, "Connection to database failed: %s",
			PQerrorMessage(conn));
		exit_nicely(conn);
	}


#ifdef BENCHMARKING
	std::chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();
#endif
	//get the features
	featureVector = calculateFeatures(conn, xorigin, yorigin, radius, k, srid);
#ifdef BENCHMARKING
	end = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsed_seconds = end - start;
	cout << "Elapsed time: " << elapsed_seconds.count() << "s\n";
#endif    


	/* close the connection to the database and cleanup */
	PQfinish(conn);

	/*now send it to python*/
	npy_intp dims[2] = { (int)featureVector.size() / 3,3 };

	double *memory = (double *)malloc(featureVector.size() * sizeof(double));
	memcpy(memory, featureVector.data(), featureVector.size() * sizeof(double)); // python should deallocate it later

	PyArrayObject* numpyArray = (PyArrayObject*)PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, memory);

	Py_INCREF(numpyArray);
	return (PyObject *)numpyArray;
}

//Table of module-level functions
static PyMethodDef pgrangeMethods[] = {

	//{module function name, implemented function name, input arg type, doc}
	{"performQuery", pgrange_performQuery, METH_VARARGS, "Perform the hard-coded query"},
	{NULL, NULL, 0, NULL}        // Sentinel
};

#if PY_MAJOR_VERSION >= 3 //Python 3.x

//Define a PyModuleDef
static struct PyModuleDef pgrangeDef =
{
	PyModuleDef_HEAD_INIT,	//m_base: always initialize with PyModuleDef_HEAD_INIT
	"pgrange",		//m_name: name of the new module
	"",			//m_doc: docstring for the module
	-1,			//m_sze: -1 means that the module does not support sub-interpreters
	pgrangeMethods,		//m_methods: a pointer to a table of module-level functions
	NULL,			//m_slots: array of slot definitions for multi-phase initialization
	NULL,			//m_traverse: a traversal function to call during GC traversal of the module object
	NULL,			//m_clear: clear function tocall during GC clearing of the module object
	NULL			//m_free: function to call during deallocation fo the module object
};
#endif

#if PY_MAJOR_VERSION >= 3 //Python 3.x

//Module initialization
PyMODINIT_FUNC PyInit_pgrange(void)
{
	PyObject *tmp = PyModule_Create(&pgrangeDef);
	import_array();
	return tmp;
}

#else //Python 2.x, working

//Module initialization
PyMODINIT_FUNC initpgrange(void)
{
	(void)Py_InitModule("pgrange", pgrangeMethods);
	import_array();

}

#endif
