#/bin/bash
CUDA_ROOT=/usr/local/cuda-10.0
TF_ROOT=/home/david/anaconda3/envs/ljs/lib/python3.7/site-packages/tensorflow
/usr/local/cuda-10.0/bin/nvcc -std=c++11 -c -o tf_grouping_g.cu -o tf_grouping_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
#TF 1.5
g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I ${TF_ROOT}/include -I ${CUDA_ROOT}/include -I ${TF_ROOT}/include/external/nsync/public -lcudart -L ${CUDA_ROOT}/lib64/ -L ${TF_ROOT} -ltensorflow_framework -O2 #-D_GLIBCXX_USE_CXX11_ABI=0
