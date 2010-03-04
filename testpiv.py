# WesLee Frisby
# Tesing PIV algorithm
"""
Algorithm Procedure:
1. 2D FFT on blocks
2. Complex conjugate multiplication
3. 2D IFFT
4. Peak Detection
"""
from PIVKernels import tran16, ccmult, maxloc

import pycuda.driver as cuda
import numpy as np
import pylab

from pycudafft import FFTPlan

class FGrid:
    def __init__(self, fx, fy, nx, ny):
        self.fx = fx
        self.fy = fy
        self.nx = nx
        self.ny = ny 
        self.tx = np.linspace(0,1,self.nx)
        self.ty = np.linspace(0,1,self.ny)
        self.x, self.y = np.meshgrid(self.tx,self.ty)
        self.grid = (np.cos(2*np.pi*self.fx*self.x)*np.cos(2*np.pi*self.fy*self.y)).astype(np.complex64)
        
class _2DFFT:
    def __init__(self,nx,ny,bsize=16):
        self.nx = np.int32(nx)
        self.ny = np.int32(ny)
        self.bsize = bsize #block size
        self.batch = nx/bsize*ny #How many 1D-FFTs?
        #
        self.fftplan = FFTPlan(self.bsize,1)
    def execute(self, data, out, transpose, tdata, reverse=False):
        self.fftplan.execute(data, batch=self.batch, inverse=reverse)
        transpose(data, out, self.nx, self.ny, np.int32(1), block=tdata.block, grid=tdata.grid)
        self.fftplan.execute(out, batch=self.batch, inverse=reverse)
        
class PIVtransposes:
    def __init__(self,nx,ny,bsize=16):
        self.nx = nx
        self.ny = ny
        self.bsize = 16
        self.block = (self.bsize,self.bsize/2,1) 
        self.grid = (self.nx/self.bsize,self.ny/self.bsize)
        self.numwind = self.grid[0]*self.grid[1]
class mydisplay:
    def __init__(self):
        self.count=0       
    def displayResults(self,res, cm=pylab.cm.jet, title='Specify a title'):
        self.count=self.count+1
        pylab.figure(self.count)
        pylab.imshow(res, cm)
        pylab.colorbar()
        pylab.title(title)
        
fx = 0
fy = 0
nx = 2**10
ny = 2**10
grid = FGrid(fx,fy,nx,ny).grid

displayResults = mydisplay().displayResults

displayResults(grid.real,title="Initial Grid") 

start = cuda.Event()
stop = cuda.Event()

ccview = np.asarray(grid).copy()
view1 = np.asarray(grid).copy()
view2 = np.asarray(grid).copy()
ifftview = np.asarray(grid).copy()

trandata = PIVtransposes(nx,ny) #Build parameters for tranpose

hostpeaks = np.zeros(trandata.numwind,np.int32) #Host peak data
gpupeaks = cuda.mem_alloc(hostpeaks.nbytes) #Device memory to store the peak result

gpuimage1 = cuda.mem_alloc(grid.nbytes) #Memory for image1 on GPU
gpuimage2 = cuda.mem_alloc(grid.nbytes) #Memory for image2 on GPU

gpuresult1 = cuda.mem_alloc(grid.nbytes) #Memory for transpose of image1 on GPU
gpuresult2 = cuda.mem_alloc(grid.nbytes) #Memory for transpose of image2 on GPU

cuda.memcpy_htod(gpuimage1, grid) #transfer grid image to device
cuda.memcpy_htod(gpuimage2, grid) #transfer the same grid to other image

plan = _2DFFT(nx,ny) #setup 2D FFT Plan

start.record()
plan.execute(gpuimage1, gpuresult1, tran16, trandata) #execute 2D FFT on image1
plan.execute(gpuimage2, gpuresult2, tran16, trandata) #execute 2D FFT on image2
stop.record()
stop.synchronize()
exec_time = stop.time_since(start)

#transfer back the results to test for correctness
cuda.memcpy_dtoh(view1, gpuresult1)
cuda.memcpy_dtoh(view2, gpuresult2)

displayResults(view1.real,title="FFT 1")
displayResults(view2.real,title="FFT 2")

start.record()
ccmult(gpuresult1, gpuresult2, np.int32(nx), np.int32(ny), block=(16,16,1), grid=trandata.grid)
stop.record()
stop.synchronize()
ccmult_time = stop.time_since(start)

cuda.memcpy_dtoh(ccview, gpuresult1)

displayResults(ccview.real,title="CCMULT Result")

start.record()
plan.execute(gpuresult1, gpuresult2, tran16, trandata, reverse=True ) #inverse FFT
stop.record()
stop.synchronize()
ifft_time = stop.time_since(start)

cuda.memcpy_dtoh(ifftview, gpuresult2)

displayResults(ifftview.real,title="IFFT Result")

start.record()
maxloc(gpuresult1, gpupeaks, block=(16,16,1), grid=trandata.grid) #peak detection
stop.record()
stop.synchronize()
mloc_time = stop.time_since(start)

cuda.memcpy_dtoh(hostpeaks, gpupeaks)

def mloc(ary):
    f = 2**16-1
    for x in range(len(ary)):
        xloc = ary[x] >> 16;
        yloc = ary[x] & f;
        print 'x',xloc,'y',yloc
        
#mloc(hostpeaks)
print "These times are in miliseconds:"
print "2-Image FFT time:",exec_time
print "CCMult time:",ccmult_time
print "IFFT time:",ifft_time
print "Maxloc time:",mloc_time
