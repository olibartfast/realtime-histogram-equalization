/*
 * histoEqCuda.h
 *
 *  Created on: Oct 1, 2016
 *      Author: x
 */

#ifndef HISTOEQCUDA_H_
#define HISTOEQCUDA_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

#define HISTOGRAM_LENGTH 256
#define BSIZE 1024
#define BSIZE2D 32

#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(-10*error); \
    }\
}

__device__ float d_p(unsigned int x, int width, int height){
    return ((float)x / (float)(width * height));
}

//Use the same clamp function you used in the Image Convolution MP
__device__ float d_clamp (float x, float start, float end){
    return min(max(x, start), end);
}

//#1
//Convert the image from RGB to GrayScale
//Implement a kernel that converts the the RGB image to GrayScale
__global__ void d_RGBtoGS(unsigned char* grayImage, unsigned char* ucharImage,
		int height, int width)
{
    int jj=blockDim.x*blockIdx.x+threadIdx.x;
    int ii=blockDim.y*blockIdx.y+threadIdx.y;

	if (ii>=height || jj>=width )
		return;
    int idx = (ii * width + jj);
    //here channels is 3

    unsigned char r = ucharImage[3 * idx];
    unsigned char g = ucharImage[3 * idx + 1];
    unsigned char b = ucharImage[3 * idx + 2];
    grayImage[idx] = (unsigned char) (0.21*r + 0.71*g + 0.07*b);
}


//#2
//Compute the histogram of  grayImage
//Implement a kernel that computes the histogram (like in the lectures) of the image.
__global__ void d_Histogram(unsigned int *histogram, unsigned char *grayImage,
		int width, int height){
    int ii=blockDim.x*blockIdx.x+threadIdx.x;
    if( ii >= width * height)
        return;
    __shared__ unsigned int smHisto[BSIZE + 1];
    if(threadIdx.x<BSIZE)
        smHisto[threadIdx.x]=0;
	__syncthreads();
    int stride=blockDim.x*gridDim.x;
    while (ii<width * height){
        atomicAdd(&(smHisto[grayImage[ii]]),1);
        ii+=stride;
        }
    __syncthreads();
    if(threadIdx.x<BSIZE)
        atomicAdd(&(histogram[threadIdx.x]), smHisto[threadIdx.x]);
}


//#3
//Compute the Comulative Distribution Function of  histogram
__global__ void d_Cdf(float* cdf, unsigned int* histogram, int width,
		int height, int bins){
	int ii=blockDim.x*blockIdx.x+threadIdx.x;
	if( ii >= BSIZE)
		return;

// A Work-Efficient Parallel Scan Kernel
//Two-phased balanced tree traversal
	__shared__ float smCdf[BSIZE + 1];
	if(ii < bins)
        smCdf[threadIdx.x] = d_p(histogram[ii], width, height);
	else smCdf[threadIdx.x] = 0;
	__syncthreads();

//Parallel Scan - Reduction Phase
	for(unsigned int stride=1; stride<BSIZE; stride*=2){
		__syncthreads();
		int index=(threadIdx.x+1)*stride*2-1;
		if(index<BSIZE)
			smCdf[index]+=smCdf[index-stride];
	}
	__syncthreads();
//Parallel Scan - Post Reduction Reverse Phase
    for (int stride = BSIZE; stride > 0; stride /= 2) {
    	__syncthreads();
        int index = (threadIdx.x+1)*stride*2 - 1;
        if(index+stride < BSIZE) {
            smCdf[index + stride] += smCdf[index];
        }
    }
    __syncthreads();
    cdf[ii] = smCdf[threadIdx.x];


}

//#4
//Compute the minimum value of the CDF
//This is a reduction operation using the min function
__global__ void d_MinOfCdf(const float* const input, float * output, int len){
	__shared__ float partialMin[BSIZE * 2 + 1];

	unsigned int t=threadIdx.x;
	unsigned int start=2*blockIdx.x*blockDim.x;

	if(start+t<len)
		partialMin[t]=input[start+t];
  	else partialMin[t]=0;

	if(start+blockDim.x+t<len)
		partialMin[blockDim.x+t]=input[start+blockDim.x+t];
	else partialMin[blockDim.x+t]=0;
	__syncthreads();

 	for(unsigned int stride = blockDim.x; stride>=1; stride>>=1)
	{

		if(t<stride)
		{
			float temp=min(partialMin[t+stride], partialMin[t]);
			__syncthreads();
			partialMin[t]=temp;
			__syncthreads();
		}
  	}

  	if (threadIdx.x == 0){
    	output[0]=partialMin[0];
	}
}

//#5
//Define the histogram equalization function
//The histogram equalization function ( correct ) remaps the cdf of the histogram
//of the image to a linear function
//and
//#6
//Apply the histogram equalization function
//Once you have implemented all of the above, then you are ready to correct the input image
__global__ void d_Hef(unsigned char* ucharImage, float *cdf, float cdfmin,
		int width, int height, int channels){
    int ii = blockDim.x*blockIdx.x+threadIdx.x;
    int jj = blockDim.y*blockIdx.y+threadIdx.y;

	if (ii >= width|| jj >= height )
		return;

	int z = (jj*width+ii)*channels;
    ucharImage[z] = d_clamp(255*(cdf[ucharImage[z]] - cdfmin)/(1.0 - cdfmin), 0, 255);
    ucharImage[z + 1] = d_clamp(255*(cdf[ucharImage[z + 1]] - cdfmin)/(1.0 - cdfmin), 0, 255);
    ucharImage[z + 2] = d_clamp(255*(cdf[ucharImage[z + 2]] - cdfmin)/(1.0 - cdfmin), 0, 255);        



}




#endif /* HISTOEQCUDA_H_ */
