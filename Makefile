CXX=g++-4.2
CFLAGS=-bundle -undefined dynamic_lookup
CLINK=-Wl,-F. -bundle -undefined dynamic_lookup
LIBS=-L/Users/wfrisby/pool/lib

BOOST_INCLUDE=-I/Users/wfrisby/pool/include/boost-1_39/
PYTHON_INCLUDE=-I/System/Library/Frameworks/Python.framework/Versions/2.6/include/python2.6
NUMPY_INCLUDE=-I/System/Library/Frameworks/Python.framework/Versions/2.6/Extras/lib/python/numpy/core/include/

INCLUDE=$(BOOST_INCLUDE) $(PYTHON_INCLUDE) $(NUMPY_INCLUDE) -I.

LINK=-lboost_python-xgcc40-mt

EXTRAS=-isysroot /Developer/SDKs/MacOSX10.6.sdk -arch i386 -m32


all:
	$(CXX) $(CFLAGS) $(INCLUDE) -c vtk_lib.c $(EXTRAS)
	$(CXX) $(CLINK) $(LIBS) $(INCLUDE) $(LINK) vtk_lib.o wrap_image_reading.cpp -o _piv.so $(EXTRAS)

clean:
	rm _piv.so
