"""
This PyCUDA script should allow for testing of kernels that will
find the location of the maximum value on a GPU.
WesLee Frisby - March 2010
"""

import pycuda.autoinit
import pycuda.driver as cuda
import numpy

from pycuda.compiler import SourceModule
# First test is with two values one positive and one negative

def return_max(a,b):
	c = a > b
	return c*a + (c^1)*b
	
	
Kernel_maxloc = """
struct {
	int loc, __padding; //Keeps 64-bit pointers aligned
	float *ptr;
};

__global__ maxloc(int *in, int *out) 
{
int blockId = 
}
"""


Kernel_maxval = """

#define SIZE_BLOCK 16
#define WIDTH 32

__global__ void maxvalue(float *in, float *out)
{
int bindex = blockIdx.x*SIZE_BLOCK + blockIdx.y*SIZE_BLOCK*WIDTH;
int index = threadIdx.x + threadIdx.y*SIZE_BLOCK;
int iodex = bindex+index;
__shared__ float smem[SIZE_BLOCK*SIZE_BLOCK+1];
smem[index] = (blockIdx.y+1)*in[iodex]; 
out[iodex] = smem[index];
}
"""

Kernel_simple_maxval = """
__global__ void maxvalue(float *in, float *out)
{
int index = threadIdx.x;
__shared__ float smem[120];

smem[index] = in[index];

__syncthreads();

for(unsigned int s=1; s < blockDim.x; s*=2) {
    int mindex = 2*s*index;
    if(mindex < blockDim.x) {
        int g = smem[mindex] > smem[mindex+s];
        smem[mindex] = g*smem[mindex] + (!g)*smem[mindex+s];
    }
    __syncthreads();
}
if(index==0) out[0] = smem[0];
}
"""


Kernel_branch_maxval = """
__global__ void maxvalue(float *in, float *out)
{
int index = threadIdx.x;
__shared__ float smem[120];

smem[index] = in[index];

__syncthreads();

for(unsigned int s=1; s < blockDim.x; s*=2) {
    int mindex = 2*s*index;
    if(mindex < blockDim.x) {
         if(smem[mindex] > smem[mindex+s]) {
            smem[mindex] = smem[mindex];
         }
         else {
            smem[mindex] = smem[mindex+s];
         }
    }
    __syncthreads();
}
if(index==0) out[0] = smem[0];
}
"""

Kernel_block_maxval = """
__global__ void maxvalue(float *in, float * out)
{
int gindex = threadIdx.x + blockDim.x*blockIdx.x;  
int tindex = threadIdx.x;
__shared__ float smem[16];

smem[tindex] = in[gindex];

__syncthreads();

for(unsigned int s=1; s < blockDim.x; s*=2) {
    int mindex = 2*s*tindex;
    if(mindex < blockDim.x) {
        int g = smem[mindex] > smem[mindex+s];
        smem[mindex] = g*smem[mindex] + (!g)*smem[mindex+s];
    }
    __syncthreads();
}

if(tindex==0) out[blockIdx.x] = smem[0];

}
"""

"""
//gindex = tindex + blockDim.x*blockIdx.x + gridDim.x*blockDim.x*blockDim.y*blockIdx.y;
int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
//gindex = xIndex + yIndex * gridDim.x*blockDim.x;
"""

Kernel_2Dblock_maxval = """
__global__ void maxvalue(float *in, float *out)
{
int tindex = threadIdx.x + blockDim.x * threadIdx.y;
int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
int gindex = xIndex + yIndex * gridDim.x*blockDim.x;

__shared__ float smem[16*16+1];
smem[tindex] = in[gindex];
__syncthreads();

int bd = blockDim.x*blockDim.y;

for(unsigned int s=1; s < bd; s*=2) {
    int mindex = 2*s*tindex;
    if(mindex < bd) {
        int g = smem[mindex] > smem[mindex+s];
        smem[mindex] = g*smem[mindex] + (!g)*smem[mindex+s];
    }
    __syncthreads();
}
if (tindex == 0) out[blockIdx.x + gridDim.x*blockIdx.y] = smem[0];
}
"""

smod = SourceModule(Kernel_2Dblock_maxval)
func = smod.get_function("maxvalue")

Kernel_2Dblock_maxloc = """
__global__ void maxloc(float * in, int * out)
{
int tindex = threadIdx.x + blockDim.x*threadIdx.y;
int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
int gindex = xIndex + yIndex * gridDim.x * blockDim.x;

__shared__ float smem[16*16];
__shared__ int mloc[16*16];
smem[tindex] = in[gindex];
mloc[tindex] = threadIdx.y | (threadIdx.x << 16);
__syncthreads();

int bd = blockDim.x*blockDim.y;

for(unsigned int z=1; z < bd; z*=2) {
    int mindex = 2*z*tindex;
    if(mindex < bd) {
        int g = smem[mindex] > smem[mindex+z];
        smem[mindex] = g*smem[mindex] + (!g)*smem[mindex+z];
        mloc[mindex] = g*mloc[mindex] + (!g)*mloc[mindex+z];
    }
    __syncthreads();
}
if (tindex == 0) out[blockIdx.x + gridDim.x*blockIdx.y] = mloc[0];
}
"""

smod = SourceModule(Kernel_2Dblock_maxloc)
func = smod.get_function("maxloc")

"""a = numpy.arange(16*16)
a = a.astype(numpy.float32)
b = numpy.zeros(16*16*4,dtype=numpy.float32)
for i in range(0,4):
    b[i*16*16:16*16*(i+1)] = a[0:16*16]

i_gpu = cuda.mem_alloc(b.nbytes)
cuda.memcpy_htod(i_gpu, b)
o_gpu = cuda.mem_alloc(b.nbytes)


func(i_gpu, o_gpu, block=(16,16,1), grid=(2,2))

cuda.memcpy_dtoh(b,o_gpu)"""


mysize = 256;
a = numpy.arange(mysize)
a = a.astype(numpy.float32)
b = numpy.zeros(16*16*4,dtype=numpy.float32)
for i in range(0,4):
    b[i*16*16:16*16*(i+1)] = a[0:16*16].copy()

i_gpu = cuda.mem_alloc(b.nbytes)
o_gpu = cuda.mem_alloc(16)
cuda.memcpy_htod(i_gpu, b)
func(i_gpu, o_gpu, block=(16,16,1), grid=(2,2))
c = numpy.ones(4,dtype=numpy.int32)
cuda.memcpy_dtoh(c,o_gpu)

i = 0
for x in range(32):
    print repr(x).rjust(2),
    for y in range(32):
        print repr(i).rjust(4),
        i = i + 1
    print
    
    
i = 0
for  x in range(32):
    for y in range(32):
        print repr(b[i]).rjust(5),
        i = i + 1
    print


mysize = 2**10*2**10
a = numpy.arange(mysize)
a = a.astype(numpy.float32)

a[0] = 800.0

i_gpu = cuda.mem_alloc(a.nbytes)
o_gpu = cuda.mem_alloc(4*4096)
cuda.memcpy_htod(i_gpu, a)
func(i_gpu, o_gpu, block=(16,16,1), grid=(64,64))
c = numpy.ones(4096,dtype=numpy.int32)
cuda.memcpy_dtoh(c,o_gpu)

def mloc(ary):
    f = 2**16-1
    for x in range(len(ary)):
        xloc = ary[x] >> 16;
        yloc = ary[x] & f;
        print 'x',xloc,'y',yloc


