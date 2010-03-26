import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import pylab

from pycudafft import FFTPlan
from cufft import CUFFTPlan

from pycuda.compiler import SourceModule

import time
import math

# Test benchmarks
# pycudafft vs cufft
#



class FGrid:
    def __init__(self,fx, fy, nx, ny):
        self.fx = fx
        self.fy = fy
        self.nx = nx
        self.ny = ny 
        self.tx = np.linspace(0,1,self.nx)
        self.ty = np.linspace(0,1,self.ny)
        self.x,self.y = np.meshgrid(self.tx,self.ty)
        self.grid = (np.cos(2*np.pi*self.fx*self.x)*np.cos(2*np.pi*self.fy*self.y)).astype(np.complex64)
"""
The PIVtransposes includes transposes specific to 
the PIV project. The transpose function should be 
converted to a templated kernel.
"""

class PIVtransposes:
    def __init__(self,nx,ny,bsize=16):
        self.nx = nx
        self.ny = ny
        #self.cache = cache
        self.bsize = 16
        self.trans()
        self.build()
    def trans(self):
        transposes = """
        __global__ void transpose16(float2 * idata, float2 *odata, int width, int height, int num)
        {
 
        __shared__ float blockx[16][16+1];
        __shared__ float blocky[16][16+1];

        int xIndex = blockIdx.x * 16 + threadIdx.x;
        int yIndex = blockIdx.y * 16 + threadIdx.y;
        int index = xIndex + yIndex*width;

           for (int i= 0; i< 16; i+=8) {
               blockx[threadIdx.y+i][threadIdx.x] = idata[index + i*width].x;
               blocky[threadIdx.y+i][threadIdx.x] = idata[index + i*width].y;
           }

        __syncthreads();

            for (int i = 0; i < 16; i+=8) {
                odata[index+i*height].x = blockx[threadIdx.x][threadIdx.y+i];
                odata[index+i*height].y = blocky[threadIdx.x][threadIdx.y+i];
            }
        }
        """
        t = SourceModule(transposes)
        self.t16 = t.get_function("transpose16")
        #self.t32 = t.get_function("transpose32")
        #self.t64 = t.get_function("transpose64")
    def build(self):
        self.block = (self.bsize,self.bsize/2,1) 
        self.grid = (self.nx/self.bsize,self.ny/self.bsize)
        
class _2DFFT:
    def __init__(self,nx,ny,bsize=16):
        self.nx = np.int32(nx)
        self.ny = np.int32(ny)
        self.bsize = bsize
        self.batch = nx/bsize*ny
        from pycudafft import FFTPlan
        self.fftplan = FFTPlan(self.bsize,1)
    def execute(self, data, out, transpose, tdata):
        self.fftplan.execute(data, batch=self.batch)
        transpose(data, out, self.nx, self.ny, np.int32(1), block=tdata.block, grid=tdata.grid)
        self.fftplan.execute(out, batch=self.batch)

rundev = cuda.Device(0)
print "Running on", rundev.name()
print "With",rundev.total_memory(),"bytes of memory"
        
fx = 3
fy = 3
for i in range(3):
    time_start = time.time()
    nx = 2**(5+i)
    ny = 2**(5+i)
    grid = FGrid(fx,fy,nx,ny).grid

    gpudata = cuda.mem_alloc(grid.nbytes) #allocate memory on device
    gpuresult = cuda.mem_alloc(grid.nbytes) #output for the transpose
    cuda.memcpy_htod(gpudata, grid) #transfer data to device

    plan = _2DFFT(nx,ny)
    trans = PIVtransposes(nx,ny)
    transfunc = trans.t16

    start = cuda.Event()
    stop = cuda.Event()

    start.record()
    plan.execute(gpudata, gpuresult, transfunc, trans)
    stop.record()
    stop.synchronize()
    t_time = stop.time_since(start)

    gridres = np.empty_like(grid)
    cuda.memcpy_dtoh(gridres,gpuresult) # transfer data from device to host
    time_past = time.time() - time_start
    
    print "2DFFT size:",nx,"x",ny,"kernel time",t_time,"ms","batch",plan.batch/16,"block size",plan.bsize
    print "Elapsed time:",time_past

def displayResults(res, count, cm=pylab.cm.jet, title='Specify a title'):
    pylab.figure(count)
    pylab.imshow(res, cm)
    pylab.colorbar()
    pylab.title(title)
    
displayResults(grid.real,1,title="Grid")
displayResults(gridres.real,2,title="FFT result")
pylab.show()


