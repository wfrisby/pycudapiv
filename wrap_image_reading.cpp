#include <Python.h>
#include <vtk_lib.h>
#include <utility>
#include <numeric>
#include <algorithm>
#include <numpy/arrayobject.h>
#include <boost/python.hpp>


//#include "ReadIM7.h"

/*namespace 
{
  static struct array_importer
  {
    array_importer()
    { import_array(); }
  } _array_importer;
}*/


using namespace boost::python;
namespace py = boost::python;


//import_array();
  
  /*void load_image(py::object filename, py::object data)
  {
      char const * file = py::extract<char const*>(filename);
      BufferType mybuffer;
      short * adata = (short*)PyArray_DATA(data.ptr());
      PyReadIM7(file,&mybuffer,adata);
      printf("The size is %d x %d x %d",mybuffer.nx,mybuffer.ny,mybuffer.nf);
  }*/
  
  void read_bin_image(char const *filename, short* buffer)
  {
      FILE * myfile = fopen(filename,"rb");
      fread((short*)buffer,sizeof(short),1024*1024,myfile);
      fclose(myfile);
  }
  
  void load_bin_image(py::object filename, py::object data)
  {
      char const * file = py::extract<char const*>(filename);
      short * adata = (short*)PyArray_DATA(data.ptr());
      read_bin_image(file,adata);
  }
  
  /* This function expects that the data object is complex valued with u,v combined*/
  void write_vtk(py::object filename, py::object data, py::object winsize)
  {
      char const * file = py::extract<char const*>(filename);
      int windsize = py::extract<int>(winsize);
      int vecsize = 1024/windsize;
      FILE * myfile = fopen(file,"w");
      float2 * vectors = (float2*)PyArray_DATA(data.ptr());
      //int nd = PyArray_NDIMS();
      //int dims = PyArray_DIMS();
      //We know the size of the images
      //so we are going to hardcode the size
      vtk_write_header(myfile,file);
      vtk_write_grid_2d(myfile, vecsize, vecsize);
      vtk_write_vector_2d(myfile, "Pass(1)", vectors, vecsize*vecsize);
      fclose(myfile);
  }
  
BOOST_PYTHON_MODULE(_piv)
{
  //py::def("load_image", load_image,
  //     (py::arg("filename"), py::arg("data")));
       
  py::def("load_bin_image", load_bin_image,
       (py::arg("filename"), py::arg("data")));

  py::def("write_vtk", write_vtk,
       (py::arg("filename"), py::arg("data"), py::arg("winsize")));
}


