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




std::string range_query(double xmin, double ymin, double xmax, double ymax)
{
   std::stringstream ss;
   ss.precision(std::numeric_limits<double>::max_digits10);

   ss << "SELECT x,y,z FROM points WHERE geom && ST_MakeEnvelope(" << xmin<< "," << ymin << "," << xmax << "," << ymax << ");";
   return(ss.str());
}

std::string knn_query(double x, double y, int k)
{

   std::stringstream ss;
   ss.precision(std::numeric_limits<double>::max_digits10);
   ss << "SELECT * FROM points order by geom <#> ST_Point(" << x << "," << y << ") LIMIT " << k << ";";
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

   char *bptr = PQgetvalue(res, i, XIndex );
   double x = double_swap(*((double*) bptr));
   return x;
}


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

    
    int N =PQntuples(res);
    std::vector<double> ret(3*N);
    int idx_x = PQfnumber(res,"X");
    int idx_y = PQfnumber(res,"Y");
    int idx_z = PQfnumber(res,"Z");
    
     for (int i=0; i < N; i++)
     {
	ret[i*3] = fieldValue(res,i,idx_x);
	ret[i*3+1] = fieldValue(res,i,idx_y);
        ret[i*3+2] = fieldValue(res,i,idx_z);
		
     }


    PQclear(res);
    return ret;
}

/* Now your specific example*/

#include<map>
#include<algorithm>
	
/* YOUR QUERY*/

std::string area_query(double x, double y, double my_128 = 128.0)
{
   return range_query(x-my_128, y-my_128, x+my_128, y+my_128);
}


std::vector<double> calculateFeatures(PGconn *conn,double xorigin = 545390.445795515,   double yorigin = 5800605.7953092)
{
   std::vector<double> ret;

   auto region = pointsFromSQL(conn,area_query( xorigin , yorigin ));
   // now, we need a function to calculate the cell coordinates for each point.
   auto homogenous = [xorigin,yorigin](double x, double y) {return std::make_pair((int) (x-xorigin), (int) (y - yorigin));};
   auto dehomogenous = [xorigin,yorigin](std::pair<int,int> p) {return std::make_pair((xorigin-p.first),(yorigin-p.second));};

   std::map<std::pair<int,int>, std::vector<int>>  cells;
   for (size_t i=0; i < region.size() / 3; i++)
   {
      auto hc = homogenous(region[i*3],region[i*3+1]);
      cells[hc].push_back(i);
   }
   auto i2z = [&region](size_t i) {return region[i*3+2];};

   
   std::vector<double> Zvalues;
   for (int ix=-64; ix<64; ix++)
   for (int iy=-64; iy<64; iy++)
   {
      Zvalues.clear(); // just for debugging.
      auto hc = std::make_pair(ix,iy); // homogenous coordinates of cell
      auto a = cells[hc];
      if (a.empty())
      {        

	auto cp = dehomogenous(hc);
       	auto q = knn_query(cp.first, cp.second, 3);	
	auto knn = pointsFromSQL(conn,q);
	Zvalues.resize(knn.size() / 3);
	for (size_t i=0; i < knn.size() / 3; i++)
	  Zvalues[i] = knn[i*3+2];

      }else{
          Zvalues.resize(a.size());
          std::transform(a.begin(), a.end(),Zvalues.begin(),i2z);
      }
      // now statistics on Zvalues
      double average = std::accumulate( std::begin( Zvalues ), std::end( Zvalues ), 0.0 ) / std::distance( std::begin( Zvalues ), std::end( Zvalues ) );

      auto minmax = std::minmax_element(std::begin( Zvalues ), std::end( Zvalues));
//      cout << ix << " " << iy << " " << Zvalues.size() << " " << average << " "
  //         << *(minmax.first) << " " << *(minmax.second) << endl;
            ret.push_back(average);
	    ret.push_back(*(minmax.first));
	    ret.push_back(*(minmax.second));
   
   }
   
   return ret;
}






// Load a dataset from a Shape File.
static PyObject *
pgrange_performQuery(PyObject *self, PyObject *args)
{
double xorigin, yorigin;
std::vector<double> featureVector;
const char *conninfo;
    if (!PyArg_ParseTuple(args, "sdd", &conninfo,&xorigin, &yorigin))
        return NULL;

    PGconn     *conn;
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
      featureVector =  calculateFeatures(conn,xorigin,yorigin);
#ifdef BENCHMARKING
    end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = end-start;
   cout << "Elapsed time: " << elapsed_seconds.count() << "s\n";
#endif    

   
    /* close the connection to the database and cleanup */
    PQfinish(conn);
    /*now send it to python*/

    npy_intp dims[2] = {(int) featureVector.size()/3,3};

   double *memory = (double *) malloc(featureVector.size() * sizeof(double));
   memcpy(memory, featureVector.data(), featureVector.size()*sizeof(double)); // python should deallocate it later

   PyArrayObject* numpyArray = (PyArrayObject*)PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, memory);

    Py_INCREF(numpyArray);    
    return (PyObject *) numpyArray;
}




static PyMethodDef pgrangeMethods[] = {
 
    {"performQuery",  pgrange_performQuery, METH_VARARGS, "Perform the hard-coded query"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};


PyMODINIT_FUNC
initpgrange(void)
{
    (void) Py_InitModule("pgrange", pgrangeMethods);
    import_array();

}
