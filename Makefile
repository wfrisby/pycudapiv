CXX=g++-4.2
CFLAGS=-Wl,-F. -bundle -undefined dynamic_lookup
LIBS=-L/Users/wfrisby/pool/lib

BOOST_INCLUDE=-I/Users/wfrisby/pool/include/boost-1_39/
PYTHON_INCLUDE=-I/System/Library/Frameworks/Python.framework/Versions/2.6/include/python2.6
NUMPY_INCLUDE=-I/System/Library/Frameworks/Python.framework/Versions/2.6/Extras/lib/python/numpy/core/include/

INCLUDE=$(BOOST_INCLUDE) $(PYTHON_INCLUDE) $(NUMPY_INCLUDE)

LINK=-lboost_python-xgcc40-mt

EXTRAS=-isysroot /Developer/SDKs/MacOSX10.6.sdk -arch i386 -m32

all:
	$(CXX) $(CFLAGS) $(LIBS) $(INCLUDE) $(LINK) wrap_image_reading.cpp -o _piv.so $(EXTRAS)

clean:
	rm _piv.so
