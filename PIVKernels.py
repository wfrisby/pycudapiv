# WesLee Frisby 
# Kernels used for PIV
import pycuda.autoinit
from pycuda.compiler import SourceModule

Kernel_2Dblock_maxloc = """
__global__ void maxloc(float2 * in, int * out)
{
int tindex = threadIdx.x + blockDim.x * threadIdx.y;
int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
int gindex = xIndex + yIndex * gridDim.x * blockDim.x;

__shared__ float smem[16*16];
__shared__ int mloc[16*16];
smem[tindex] = in[gindex].x;
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

transpose16 = """
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

transpose16new = """
__global__ void transpose16(float2 * idata, float2 * odata, int width, int height, int num)
{

__shared__ float blockx[16][16+1];
__shared__ float blocky[16][16+1];

int xIndex = blockIdx.x * 16 + threadIdx.x;
int yIndex = blockIdx.y * 16 + threadIdx.y;
int index = xIndex + yIndex*width;
   
       blockx[threadIdx.y][threadIdx.x] = idata[index].x;
       blocky[threadIdx.y][threadIdx.x] = idata[index].y;
       blockx[threadIdx.y+8][threadIdx.x] = idata[index + (width<<3)].x;
       blocky[threadIdx.y+8][threadIdx.x] = idata[index + (width<<3)].y;

__syncthreads();

        odata[index].x = blockx[threadIdx.x][threadIdx.y];
        odata[index].y = blocky[threadIdx.x][threadIdx.y];
        odata[index+(height<<3)].x = blockx[threadIdx.x][threadIdx.y+8];
        odata[index+(height<<3)].y = blocky[threadIdx.x][threadIdx.y+8];
}
"""

transpose16new_slow = """
__global__ void transpose16(float2 * idata, float2 * odata, int width, int height, int num)
{

__shared__ float blockx[16][16+1];
__shared__ float blocky[16][16+1];

int xIndex = blockIdx.x * 16 + threadIdx.x;
int yIndex = blockIdx.y * 16 + threadIdx.y;
int index = xIndex + yIndex*width;

   blockx[threadIdx.y][threadIdx.x] = idata[index].x;
   blocky[threadIdx.y][threadIdx.x] = idata[index].y;

__syncthreads();

   odata[index].x = blockx[threadIdx.x][threadIdx.y];
   odata[index].y = blocky[threadIdx.x][threadIdx.y];

}
"""

ComplexConjMultiplication = """

#define BLOCK_DIM 16

/**
 * Complex multiply of float2 numbers, one of which is conjugated before multiplicaiton
 *
 * @param f - First complex point -- This point will be normal.
 * @param g - Second complex point -- This point will be conjugated.
 * Output - The result is returned with h;
 *
 **/

 __device__ float2 CmplxConjMult(float2 f, float2 g)
{
  float2 h;
  //Conju Complex Multiplication
  //The product (a+bi)*(c-di)
  //k1 = c*(a+b)
  //k2 = a*(-d-c)
  //k3 = b*(c-d)
  //Real part = k1-k3
  //Imag part = k1+k2

  float k1 = g.x*(f.x + f.y);
  float k2 = f.x*(-g.y-g.x);
  float k3 = f.y*(g.x-g.y);
  h.x = k1 - k3;
  h.y = k1 + k2;
  return h;
}

/**
 * Pointwise Conjugate multiplication of 2-2D complex floating point arrays.
 *
 * @param float2 f - input/output 2D array that will store the result.
 * @param float2 g - input 2D array that will be conjugated during multiplication.
 * @param int size_x - the width of the 2D arrays
 * @param int size_y - the height of the 2D arrays
 **/

__global__ void PointWiseConjMult(float2 * f, float2 * g, int size_x, int size_y)
{
  //Shared memory which will store sections of the arrays before multiplication.
  __shared__ float2 fsection[(BLOCK_DIM+1)*BLOCK_DIM];
  __shared__ float2 gsection[(BLOCK_DIM+1)*BLOCK_DIM];

  unsigned int xBlock = __umul24(BLOCK_DIM, blockIdx.x);
  unsigned int yBlock = __umul24(BLOCK_DIM, blockIdx.y);
  unsigned int xIndex = xBlock + threadIdx.x;
  unsigned int yIndex = yBlock + threadIdx.y;

  unsigned int index_in = __umul24(size_x, yIndex) + xIndex;
  unsigned int index_section = __umul24(threadIdx.y, BLOCK_DIM+1) + threadIdx.x;

  // Load the shared variables with global data from f
  fsection[index_section].x = f[index_in].x;
  fsection[index_section].y = f[index_in].y;

  // Load the shared varibles with global data from g
  gsection[index_section].x = g[index_in].x;
  gsection[index_section].y = g[index_in].y;

  __syncthreads();

  //Perform the multiplication
  // Faster with function call. Memory value lookups seem slow.
  //This has bank conflicts...
  //float2 h;
  //float2 f = fsection[index_section];
  //float2 g = gsection[index_section];
  //float k1 = g.x*(f.x + f.y);
  //float k2 = f.x*(-g.y-g.x);
  //float k3 = f.y*(g.x-g.y);
  //h.x = k1 - k3;
  //h.y = k1 + k2;
  fsection[index_section] = CmplxConjMult(fsection[index_section],gsection[index_section]);

  __syncthreads();

  //write back to global memory
  f[index_in].x = fsection[index_section].x;
  f[index_in].y = fsection[index_section].y;
}
"""

newComplexConjMultiplication = """


/**
 * Pointwise Conjugate multiplication of 2-2D complex floating point arrays.
 *
 * @param float2 f - input/output 2D array that will store the result.
 * @param float2 g - input 2D array that will be conjugated during multiplication.
 * @param int size_x - the width of the 2D arrays
 * @param int size_y - the height of the 2D arrays
 **/

__global__ void PointWiseConjMult(float2 * f, float2 * g, int size_x, int size_y)
{
  //Shared memory which will store sections of the arrays before multiplication.
  
  unsigned int xBlock = gridDim.x*blockIdx.x;
  unsigned int yBlock = gridDim.y*blockIdx.y;
  unsigned int xIndex = xBlock + threadIdx.x;
  unsigned int yIndex = yBlock + threadIdx.y;

  unsigned int index_in = size_x*yIndex + xIndex;

  // Load the shared variables with global data from f
  //fsection[index_section].x = f[index_in].x*g[index_in].x+f[index_in].y*g[index_in].y;
  //fsection[index_section].y = f[index_in].y*g[index_in].x-f[index_in].x*g[index_in].y;

  float2 nf;
  float2 ng;
  nf = f[index_in];
  ng = g[index_in];
  
  //fsection[index_section].x = nf.x*ng.x+nf.y*ng.y;
  //fsection[index_section].y = nf.y*ng.x-nf.x*ng.y;
  
  f[index_in].x = nf.x*ng.x+nf.y*ng.y;
  f[index_in].y = nf.y*ng.x-nf.x*ng.y;

  //(fx+fyj)*(gx-gyj)
  // x = fxgx + fygy
  // y = fygx - fxgy

  //__syncthreads();

  //write back to global memory
  //f[index_in].x = fsection[index_section].x;
  //f[index_in].y = fsection[index_section].y;
}
"""

smod1 = SourceModule(Kernel_2Dblock_maxloc)
maxloc = smod1.get_function("maxloc")

smod2 = SourceModule(ComplexConjMultiplication)
ccmult = smod2.get_function("PointWiseConjMult")

smod3 = SourceModule(transpose16new)
tran16 = smod3.get_function("transpose16")

