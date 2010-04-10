# Derived from a test case by Chris Heuser
# Also see FAQ about PyCUDA and threads.

import pycuda
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import threading
import numpy

class GPUThread(threading.Thread):
    def __init__(self, number, some_array):
        threading.Thread.__init__(self)

        self.number = number
        self.some_array = some_array

    def run(self):
        
        self.dev = cuda.Device(self.number)
        self.ctx = self.dev.make_context()
        self.array_gpu = cuda.mem_alloc(self.some_array.nbytes)
        output_array = numpy.zeros((1,512),dtype='float32')
        output_array_gpu = cuda.mem_alloc(output_array.nbytes)
            
        for x in range(5):
            cuda.memcpy_htod(self.array_gpu, self.some_array)
            test_kernel(self.array_gpu,output_array,output_array_gpu,x)
            print x,'is done on', self.number
            print output_array[0,0:10]
        print "successful exit from thread %d" % self.number
        self.ctx.pop()

        del self.array_gpu
        del self.ctx

def test_kernel(input_array_gpu,output_array,output_array_gpu,val):
    mod = SourceModule("""
        __global__ void f(float * out, float * in)
        {
            int idx = threadIdx.x;
            out[idx] = in[idx] + %s;
        }
        """%str(val+5))
    func = mod.get_function("f")

    func(output_array_gpu,
          input_array_gpu,
          block=(512,1,1))
    cuda.memcpy_dtoh(output_array, output_array_gpu)

cuda.init()
some_array = numpy.ones((1,512), dtype=numpy.float32)
num = cuda.Device.count()

gpu_thread_list = []
for i in range(num):
    gpu_thread = GPUThread(i, some_array)
    gpu_thread.start()
    gpu_thread_list.append(gpu_thread)
