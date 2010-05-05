# WesLee Frisby 
# Kernels used for PIV
import pycuda.autoinit
from pycuda.compiler import SourceModule

Kernel_2Dblock_maxloc = """
struct short2_ {
    signed short x;
    signed short y;
};

//typedef struct short2_ short2;

union shorty_ {
    int itemp;
    struct short2_ myshort;
};
typedef union shorty_ myshort;

//__global__ void maxloc(float2 * in, int * out)
__global__ void maxloc(float2 * in, float2 * out)
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
if (tindex == 0) {
	myshort temp;
    temp.itemp = mloc[0];
    temp.itemp+=0x00080008; //FFT shift the maxloc
    temp.itemp&=0x00EF00EF; //mod 16 on both arrays
    //int b1,b2;
    //b1 = temp.myshort.x<0x0008;
    //b2 = temp.myshort.y<0x0008;
    //temp.myshort.x=(temp.myshort.x-0x0008)*b1 + (!b1)*(0x0008-temp.myshort.x);
    //temp.myshort.y=(temp.myshort.y-0x0008)*b2 + (!b2)*(0x0008-temp.myshort.y); //-windowsize/2 -8 -8
    //temp.myshort.x=temp.myshort.x-0x0008;
    //temp.myshort.y=0x0008-temp.myshort.y;
	out[blockIdx.x + gridDim.x*blockIdx.y].x = (float)(temp.myshort.x-0x0008);//temp.itemp;
    out[blockIdx.x + gridDim.x*blockIdx.y].y = (float)(0x0008-temp.myshort.y);
}
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
ComplexConjMultiplicationStry = """

#define BLOCK_DIM_X 32
#define BLOCK_DIM_Y 16

 __device__ float2 CmplxConjMult(float2 f, float2 g)
{
  float2 h;

  float k1 = g.x*(f.x + f.y);
  float k2 = f.x*(-g.y-g.x);
  float k3 = f.y*(g.x-g.y);
  h.x = k1 - k3;
  h.y = k1 + k2;
  return h;
}

__global__ void PointWiseConjMult(float2 * f, float2 * g, int size_x, int size_y)
{
  //Shared memory which will store sections of the arrays before multiplication.
  __shared__ float2 fsection[(BLOCK_DIM_X+1)*BLOCK_DIM_Y];
  __shared__ float2 gsection[(BLOCK_DIM_X+1)*BLOCK_DIM_Y];

  unsigned int xBlock = BLOCK_DIM_X*blockIdx.x;
  unsigned int yBlock = BLOCK_DIM_Y*blockIdx.y;
  unsigned int xIndex = xBlock + threadIdx.x;
  unsigned int yIndex = yBlock + threadIdx.y;

  unsigned int index_in = size_x*yIndex + xIndex;
  unsigned int index_section = threadIdx.y*(BLOCK_DIM_X+1) + threadIdx.x;

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
  float2 k;
  float2 h = fsection[index_section];
  float2 j = gsection[index_section];
  //float k1 = j.x*(h.x + h.y);
  //float k2 = h.x*(-j.y-j.x);
  //float k3 = h.y*(j.x-j.y);
  //k.x = k1 - k3;
  //k.x = j.x*(h.x + h.y) - h.y*(j.x-j.y);
  //k.y = k1 + k2;
  //k.y = j.x*(h.x + h.y) + h.x*(-j.y-j.x);
  fsection[index_section].x = h.x*j.x+h.y*j.y;
  fsection[index_section].y = h.y*j.x-h.x*j.y; //CmplxConjMult(fsection[index_section],gsection[index_section]);

  __syncthreads();

  //write back to global memory
  f[index_in].x = fsection[index_section].x;
  f[index_in].y = fsection[index_section].y;
}
"""

average_kernel = """
__global__ void average(float2 * in, float2 * out)
{
__shared__ float2 smem[16][16+1];

//int tindex = threadIdx.x + blockDim.x * threadIdx.y;
int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
int gindex = xIndex + yIndex * gridDim.x * blockDim.x;

//Load in the dang memory
smem[threadIdx.y][threadIdx.x].x = in[gindex].x;
smem[threadIdx.y][threadIdx.x].y = in[gindex].y;

__syncthreads();
//0 && threadIdx.x < 15 && threadIdx.y < 15
if (threadIdx.x > 0 && threadIdx.y > 0 && threadIdx.x < 15 && threadIdx.y < 15) {
    //Only compute on the inner vectors for their 8 neighbors
    smem[threadIdx.y][threadIdx.x].x = (smem[threadIdx.y][threadIdx.x+1].x +
                                      smem[threadIdx.y][threadIdx.x-1].x + 
                                      smem[threadIdx.y+1][threadIdx.x].x +
                                      smem[threadIdx.y-1][threadIdx.x].x +
                                      smem[threadIdx.y+1][threadIdx.x+1].x +
                                      smem[threadIdx.y+1][threadIdx.x-1].x +
                                      smem[threadIdx.y-1][threadIdx.x+1].x +
                                      smem[threadIdx.y-1][threadIdx.x-1].x)/8.0f;
                                      
    smem[threadIdx.y][threadIdx.x].y = (smem[threadIdx.y][threadIdx.x+1].y +
                                      smem[threadIdx.y][threadIdx.x-1].y + 
                                      smem[threadIdx.y+1][threadIdx.x].y +
                                      smem[threadIdx.y-1][threadIdx.x].y +
                                      smem[threadIdx.y+1][threadIdx.x+1].y +
                                      smem[threadIdx.y+1][threadIdx.x-1].y +
                                      smem[threadIdx.y-1][threadIdx.x+1].y +
                                      smem[threadIdx.y-1][threadIdx.x-1].y)/8.0f;
    __syncthreads();
}
//Output the dang memory
out[gindex].x = smem[threadIdx.y][threadIdx.x].x;
out[gindex].y = smem[threadIdx.y][threadIdx.x].y;

}
"""

kernel_max_32 = """
struct short2_ {
    signed short x;
    signed short y;
};

union shorty_ {
    int itemp;
    struct short2_ myshort;
};
typedef union shorty_ myshort;

__global__ void max_32(float2 *in, float2 *out)
{
int tindex = threadIdx.x + blockDim.x * threadIdx.y;
int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
int yIndex; //= blockIdx.y * blockDim.y + threadIdx.y;
int gindex; //= xIndex + yIndex * gridDim.x * blockDim.x;

__shared__ float smem[32*8];
__shared__ int mloc[32*8];

float max_value = 0;
int max_loc = 0;
int bd = blockDim.x*blockDim.y;

//Compute max in temporary region. Store the results in local varible current_max
for(unsigned int i=0; i<4; i++) {
    yIndex = blockIdx.y * blockDim.y * 4 + threadIdx.y + 8*i;
    gindex = xIndex + yIndex * gridDim.x * blockDim.x;
    smem[tindex] = in[gindex].x;
    mloc[tindex] = (threadIdx.y + 8*i) | (threadIdx.x << 16);
    
    __syncthreads(); //Wait while all input is loaded

    for(unsigned int z=1; z < bd; z*=2) {
        int mindex = 2*z*tindex;
        if(mindex < bd) {
            int g = smem[mindex] > smem[mindex+z];
            smem[mindex] = g*smem[mindex] + (!g)*smem[mindex+z];
            mloc[mindex] = g*mloc[mindex] + (!g)*mloc[mindex+z];
        }
        __syncthreads();
    }
    if(tindex==0) { //It's only one thread checking
        if(max_value<smem[0]) {
            max_loc = mloc[0];
            max_value = smem[0];
        }
    }
}

if(tindex==0) {
    myshort temp;
    temp.itemp = max_loc;
    temp.itemp+=0x00100010; //FFT shift the maxloc
    temp.itemp&=0x001F001F; //mod 32 on both arrays
    
    out[blockIdx.x + gridDim.x*blockIdx.y].x = (float)(0x00010-temp.myshort.x);
    out[blockIdx.x + gridDim.x*blockIdx.y].y = (float)(temp.myshort.y-0x0010);
}
}
"""

kernel_max_64 = """
struct short2_ {
    signed short x;
    signed short y;
};

union shorty_ {
    int itemp;
    struct short2_ myshort;
};
typedef union shorty_ myshort;

__global__ void max_64(float2 *in, float2 *out)
{
int tindex = threadIdx.x + blockDim.x * threadIdx.y;
int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
int yIndex; //= blockIdx.y * blockDim.y + threadIdx.y;
int gindex; //= xIndex + yIndex * gridDim.x * blockDim.x;

__shared__ float smem[64*8];
__shared__ int mloc[64*8];

float max_value = 0;
int max_loc = 0;
int bd = blockDim.x*blockDim.y;

//Compute max in temporary region. Store the results in local varible current_max
for(unsigned int i=0; i<8; i++) {
    yIndex = blockIdx.y * blockDim.y * 8 + threadIdx.y + 8*i;
    gindex = xIndex + yIndex * gridDim.x * blockDim.x;
    smem[tindex] = in[gindex].x;
    mloc[tindex] = (threadIdx.y + 8*i) | (threadIdx.x << 16);
    
    __syncthreads(); //Wait while all input is loaded

    for(unsigned int z=1; z < bd; z*=2) {
        int mindex = 2*z*tindex;
        if(mindex < bd) {
            int g = smem[mindex] > smem[mindex+z];
            smem[mindex] = g*smem[mindex] + (!g)*smem[mindex+z];
            mloc[mindex] = g*mloc[mindex] + (!g)*mloc[mindex+z];
        }
        __syncthreads();
    }
    if(tindex==0) { //It's only one thread checking
        if(max_value<smem[0]) {
            max_loc = mloc[0];
            max_value = smem[0];
        }
    }
}

if(tindex==0) {
    myshort temp;
    temp.itemp = max_loc;
    temp.itemp+=0x00200020; //FFT shift the maxloc
    temp.itemp&=0x003F003F; //mod 64 on both arrays
    
    out[blockIdx.x + gridDim.x*blockIdx.y].x = (float)(0x00020-temp.myshort.x);
    out[blockIdx.x + gridDim.x*blockIdx.y].y = (float)(temp.myshort.y-0x0020);
}
}
"""

transpose32= """
/*
* Transpose 32x32
* TILE_DIM = 32x32
* BLOCK_ROWS = 8
* Threads = 32x8
* Grid = size_x/32 x size_y/32
*/
#define TILE_DIM 32
#define BLOCK_ROWS 8
__global__ void transpose32(float2 * idata, float2 *odata, int width, int height, int num)
{

__shared__ float blockx[TILE_DIM][TILE_DIM+1];
__shared__ float blocky[TILE_DIM][TILE_DIM+1];

int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
int index = xIndex + yIndex*width;

   for (int i= 0; i< TILE_DIM; i+=BLOCK_ROWS) {
        blockx[threadIdx.y+i][threadIdx.x] = idata[index + i*width].x;
        blocky[threadIdx.y+i][threadIdx.x] = idata[index + i*width].y;
   }

__syncthreads();

   for (int i = 0; i < TILE_DIM; i+=BLOCK_ROWS) {
        odata[index+i*height].x = blockx[threadIdx.x][threadIdx.y+i];
        odata[index+i*height].y = blocky[threadIdx.x][threadIdx.y+i];
   }
}
"""

transpose64="""
/*
 * Transpose 64x64
 * TILE_DIM = 32
 * BLOCK_ROWS 8
 * TILE_64 = 64
 * TILE_8 = 8
 * dim3 blocks((*nx)/TILE_64,(*ny)/TILE_64);
 * dim3 threads(TILE_DIM, BLOCK_ROWS);
 *
 */
 #define TILE_DIM 32
 #define BLOCK_ROWS 8
 #define TILE_64 64
 #define TILE_8 8
 
__global__ void transpose64(float2 * idata, float2 * odata, int width, int height, int num)
{

__shared__ float blockx[TILE_DIM][TILE_DIM+1];
__shared__ float blocky[TILE_DIM][TILE_DIM+1];

//Loop over the four tiles
 for (int b = 0; b < 4; b++) {
   int offset_inx_outy = b&1; ///* b % 2
   int offset_iny_outx = b>>1; ///* b / 2
   int xIndex = blockIdx.x*TILE_64 + offset_inx_outy* TILE_DIM + threadIdx.x;
   int yIndex = blockIdx.y*TILE_64 + offset_iny_outx* TILE_DIM + threadIdx.y;
   int index = xIndex + yIndex*width;

   for (int i = 0; i < TILE_DIM; i+=BLOCK_ROWS) {
     blockx[threadIdx.y+i][threadIdx.x] = idata[index + i*width].x;
     blocky[threadIdx.y+i][threadIdx.x] = idata[index + i*width].y;
   }

   int xIndex2 = blockIdx.x*TILE_64 + offset_iny_outx*TILE_DIM + threadIdx.x;
   int yIndex2 = blockIdx.y*TILE_64 + offset_inx_outy*TILE_DIM + threadIdx.y;
   int index_out = xIndex2 + yIndex2*width;

   __syncthreads();

   for (int i = 0; i < TILE_DIM; i+=BLOCK_ROWS) {
     odata[index_out+i*height].x = blockx[threadIdx.x][threadIdx.y+i];
     odata[index_out+i*height].y = blocky[threadIdx.x][threadIdx.y+i];
   }
 }
}
"""

toComplex= """
/* This kernel converts data to a single precision complex valued datatype.
 * The kernel should be optimized to convert the data at rates near the 80GB/s
 * memory bandwidth. 
 * Optimul peak performance:
 * # of bits in memory bus * 2 (MHz of DRAM) / 8 = GB/s
 * Typical performance:
 *   Near 65GB/s on a fast machine
 *   On my laptop closer to 9.2GB/s :(
 */
//typedef unsigned short int Word;
#define TILE_DIM 32
#define BLOCK_ROWS 8

struct short2_ {
    signed short x;
    signed short y;
};

union shorty_ {
    int itemp;
    struct short2_ myshort;
};
typedef union shorty_ myshort;

__global__ void toComplex(float2 * outdata, myshort * indata, int width) //int sizex, int sizey) 
{
  __shared__ myshort data[TILE_DIM][TILE_DIM+1]; //The +1 is to avoid shared memory bank conflicts
  int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
  int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

  int index = yIndex*width + xIndex;
  int index2;
  int i;
  for (i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
        data[threadIdx.y+i][threadIdx.x] = indata[index+i*width];
  }
  
  __syncthreads();
  
  xIndex = blockIdx.x * TILE_DIM*2 + threadIdx.x*2;
  yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
  index =  yIndex*width*2 + xIndex;
  index2 = index+1;
  for (i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
        outdata[index+i*width*2].x = (float)data[threadIdx.y+i][threadIdx.x].myshort.x;
        outdata[index+i*width*2].y = 0.0f;
        outdata[index2+i*width*2].x = (float)data[threadIdx.y+i][threadIdx.x].myshort.y;
        outdata[index2+i*width*2].y = 0.0f;
  }
}
"""


smod1 = SourceModule(kernel_max_64)
maxloc = smod1.get_function("max_64")

smod2 = SourceModule(ComplexConjMultiplication)
ccmult = smod2.get_function("PointWiseConjMult")

smod3 = SourceModule(transpose64)
tran16 = smod3.get_function("transpose64")

smod4 = SourceModule(average_kernel)
average = smod4.get_function("average")

smod5 = SourceModule(toComplex)
toComplex = smod5.get_function("toComplex")
