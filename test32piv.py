# WesLee Frisby
# Tesing PIV algorithm
"""
Algorithm Procedure:
1. 2D FFT on blocks
2. Complex conjugate multiplication
3. 2D IFFT
4. Peak Detection
"""

"""
GTX 295
- Memory Clock - 999MHz
- 448bits
Maxbandwidth= 999*2*448/8 = 111.8 GB/s

Tesla
- Memory Clock - 800MHz
- 512bit
Maxbandwidth = 800*2*512/8 = 102.4 GB/s
"""

from PIVKernels import tran16, ccmult, maxloc, average

import pycuda.driver as cuda
import numpy as np
import pylab
import _piv as pivim

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
        self.grid = self.grid*0;
        
class _2DFFT:
    def __init__(self,nx,ny,bsize=16):
        self.nx = np.int32(nx)
        self.ny = np.int32(ny)
        self.bsize = bsize #block size
        self.batch = nx/bsize*ny #How many 1D-FFTs?
        self.fftplan = FFTPlan(self.bsize,1)
    def execute(self, data, out, transpose, tdata, reverse=False):
        self.fftplan.execute(data, batch=self.batch, inverse=reverse)
        transpose(data, out, self.nx, self.ny, np.int32(1), block=tdata.block, grid=tdata.grid)
        self.fftplan.execute(out, data, batch=self.batch, inverse=reverse)
        transpose(data, out, self.nx, self.ny, np.int32(1), block=tdata.block, grid=tdata.grid)
        
class PIVtransposes:
    def __init__(self,nx,ny,bsize=16):
        self.nx = nx
        self.ny = ny
        self.bsize = bsize
        if(bsize==16):
            self.block = (self.bsize,self.bsize/2,1) 
        if(bsize==32):
            self.block = (self.bsize,self.bsize/4,1)
            print "32"
        self.grid = (self.nx/self.bsize,self.ny/self.bsize)
        self.numwind = self.grid[0]*self.grid[1]
        self.shape = (self.grid[0],self.grid[1])
        
class mydisplay:
    def __init__(self,display=True):
        self.count=0
	self.display=display       
    def displayResults(self,res, cm=pylab.cm.gray, title='Specify a title'):
        if self.display:
		self.count=self.count+1
        	pylab.figure(self.count)
        	pylab.imshow(res, cm, interpolation='nearest')
        	pylab.colorbar()
        	pylab.title(title)
        
fx = 0
fy = 0
nx = 2**10
ny = 2**10

g1 = cuda.pagelocked_empty(1024*1024,'int16')
g2 = cuda.pagelocked_empty(1024*1024,'int16')

pivim.load_bin_image("frame1.bin",g1)
pivim.load_bin_image("frame2.bin",g2)

grid = g1.astype('complex64').reshape(1024,1024)
grid2 = g2.astype('complex64').reshape(1024,1024) 

displayResults = mydisplay(display=True).displayResults

displayResults(grid.real,title="Initial Grid")
displayResults(grid2.real,title="Shifted Grid")

start = cuda.Event()
stop = cuda.Event()

ccview = np.asarray(grid).copy()
view1 = np.asarray(grid).copy()
view2 = np.asarray(grid).copy()
ifftview = np.asarray(grid).copy()

trandata = PIVtransposes(nx,ny,bsize=32) #Build parameters for tranpose

hostpeaks = np.zeros(trandata.shape,np.complex64) #Host peak data
gpupeaks = cuda.mem_alloc(hostpeaks.nbytes) #Device memory to store the peak result

gpuimage1 = cuda.mem_alloc(grid.nbytes) #Memory for image1 on GPU
gpuimage2 = cuda.mem_alloc(grid.nbytes) #Memory for image2 on GPU

gpuresult1 = cuda.mem_alloc(grid.nbytes) #Memory for transpose of image1 on GPU
gpuresult2 = cuda.mem_alloc(grid.nbytes) #Memory for transpose of image2 on GPU

cuda.memcpy_htod(gpuimage1, grid) #transfer grid image to device
cuda.memcpy_htod(gpuimage2, grid2) #transfer the shifted grid to other image

plan = _2DFFT(nx,ny,bsize=32) #setup 2D FFT Plan

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
ccmult(gpuresult1, gpuresult2, np.int32(nx), np.int32(ny), block=(16,16,1), grid=(nx/16,ny/16))
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
maxloc(gpuresult1, gpupeaks, block=(32,8,1), grid=trandata.grid) #peak detection
stop.record()
stop.synchronize()
mloc_time = stop.time_since(start)

#start.record()
#average(gpupeaks, gpupeaks, block=(16,16,1), grid=(4,4))
#stop.record()
#stop.synchronize()
#avg_time = stop.time_since(start)


cuda.memcpy_dtoh(hostpeaks, gpupeaks)
#U = cuda.pagelocked_empty(64*64,'float32')
#V = cuda.pagelocked_empty(64*64,'float32')
#U = (hostpeaks.real).copy()
#V = (hostpeaks.imag).copy()

#def mloc(ary,u,v):
#    f = 2**16-1
#    for x in range(len(ary)):
#        xloc = (ary[x] >> 16) & f;
#        xloc = (xloc + 2**15) % 2**16 - 2**15
#        u[x]=float(xloc)
#        yloc = ary[x] & f;
#        yloc = (yloc + 2**15) % 2**16 - 2**15
#        v[x]=float(yloc)
#        print 'x',xloc,'y',yloc
        
#type = np.dtype((np.int32, {'x':(np.int16,0), 'y':(np.int16,2)}))

#mloc(hostpeaks,U,V)
print "These times are in miliseconds:"
print "2-Image FFT time:",exec_time
print "CCMult time:",ccmult_time
print "IFFT time:",ifft_time
print "Maxloc time:",mloc_time
#print "Average time:",avg_time


""" def fftshift2(data,size=16):
    newdata = np.zeros((size,size),dtype=np.float32);
    #Swap 1st and 3rd quad
    newdata[size/2:size,0:size/2] = data[0:size/2,size/2:size]
    newdata[0:size/2,size/2:size] = data[size/2:size,0:size/2]
    #quad 2 with 4
    newdata[0:size/2,0:size/2] = data[size/2:size,size/2:size]
    #quad 4 with 2
    newdata[size/2:size,size/2:size] = data[0:size/2,0:size/2]
    return newdata
"""
    
#result = fftshift2(ifftview)
#displayResults(result,title="Shifted results")
#testcc = view1*view2.conj()
#a[size/2:size,size/2:size] = view1[0:size/2,0:size/2]
#pylab.figure(10);
#pylab.quiver(U.reshape(64,64),V.reshape(64,64))
pylab.figure(11);


#gU = cuda.mem_alloc(U.nbytes)
#gV = cuda.mem_alloc(V.nbytes)

#cuda.memcpy_htod(gU, U)
#cuda.memcpy_htod(gV, V)

#average(gU, gU, block=(16,16,1), grid=(4,4))
#average(gV, gV, block=(16,16,1), grid=(4,4))

#Un = cuda.pagelocked_empty(64*64,'float32')
#Vn = cuda.pagelocked_empty(64*64,'float32')

#cuda.memcpy_dtoh(Un, gU)
#cuda.memcpy_dtoh(Vn, gV)
cvec = np.arange(32)
cvec = cvec[::-1]

pylab.quiver(cvec, cvec, hostpeaks.real,hostpeaks.imag)

pylab.show()
