nvcc -DTHRUST_IGNORE_CUB_VERSION_CHECK -dc -gencode arch=compute_61,code=sm_61 -gencode arch=compute_61,code=compute_61 -gencode arch=compute_61,code=sm_70  -g -O3 -w  -I../../rt/include -I../../rt/include/mgpu/include  -I../../rt/include/cub kernel.cu  -o kernel.o
nvcc -DTHRUST_IGNORE_CUB_VERSION_CHECK -g -g -O3 -w -gencode arch=compute_61,code=sm_61 -gencode arch=compute_61,code=compute_61 -gencode arch=compute_61,code=sm_70   -I../../rt/include -I../../rt/include/mgpu/include  -I../../rt/include/cub -L../../rt/lib  -o pr_our skel.o kernel.o support.o ../../skelapp/mgpucontext.o ../../skelapp/mgpuutil.o -lggrt -lcurand -lcudadevrt -lz
cp pr_our ../../bin

