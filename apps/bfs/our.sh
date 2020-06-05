nvcc -dc -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_61,code=sm_70  -g -O3 -w  -I../../rt/include -I../../rt/include/mgpu/include -I../../rt/include/cub ../../skelapp/skel.cu -o skel.o
nvcc -dc -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_61,code=sm_70   -g -O3 -w  -I../../rt/include -I../../rt/include/mgpu/include -I../../rt/include/cub kernel.cu  -o kernel.o
nvcc -dc -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_61,code=sm_70  -g -O3 -w  -I../../rt/include -I../../rt/include/mgpu/include -I../../rt/include/cub support.cu  -o support.o
nvcc -dc -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_61,code=sm_70 -g -O3 -w  -I../../rt/include -I../../rt/include/mgpu/include -I../../rt/include/cub ../../rt/include/mgpu/src/mgpucontext.cu -o ../../skelapp/mgpucontext.o
nvcc -dc -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_61,code=sm_70 -g -O3 -w  -I../../rt/include -I../../rt/include/mgpu/include -I../../rt/include/cub ../../rt/include/mgpu/src/mgpuutil.cpp -o ../../skelapp/mgpuutil.o
nvcc -g -g -O3 -w -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_61,code=sm_70   -I../../rt/include -I../../rt/include/mgpu/include -I../../rt/include/cub -L../../rt/lib  -o bfs_our skel.o kernel.o support.o ../../skelapp/mgpucontext.o ../../skelapp/mgpuutil.o -lggrt -lcurand -lcudadevrt -lz
cp bfs ../../bin

