
#include <iostream>


//OpenCV include file
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "histoEqCpu.h"
#include "histoEqCuda.h"


void histogramEqualizationOnHost(cv::Mat& outputImage, cv::Mat& inputImage,
    unsigned char* grayImage);
void histogramEqualizationOnDevice(cv::Mat& outputImage, cv::Mat& inputImage, 
    unsigned char* d_ucharImage, unsigned char* d_grayImage,  unsigned int* d_histogram, float *d_cdf);

static time_t start, time_last_cycle;
time_t end;
const float Ttime = 1000.0;

void compute_fps(int cnt, std::string myOutString, cv::Mat output_frame)
{
    //input parameters settings for cv::putText function 
    //string input parameter

    std::stringstream sstmMyOutS;
    sstmMyOutS<<myOutString;

    //point
    cv::Point myOutStringPoint;
    myOutStringPoint.x = 15;
    myOutStringPoint.y = 425;

    //font parameters
    int fontFace = cv::FONT_HERSHEY_COMPLEX_SMALL;
    double fontScale = 1.0;
    int red = 0, green = 0, blue = 255;
    cv::Scalar fontColor = cv::Scalar(blue, green, red);
    int fontThickness = 1;

    //put text on the output image
    cv::putText(output_frame, sstmMyOutS.str(), myOutStringPoint, fontFace, fontScale, fontColor, fontThickness,cv::LINE_AA,false);
  
    //fps computation
    //string parameter
    std::stringstream sstmFps; 
    sstmFps.precision(1);    
    float avg_fps_val = cnt*1000.0/difftime(end=clock(),start)*Ttime;
    float fps_val = (1000.0/difftime(end,time_last_cycle))*Ttime;
    sstmFps<< std::fixed << "avg fps "<<avg_fps_val<<"   fps "<<fps_val;

    //point
    cv::Point myFpsPoint;
    myFpsPoint.x = 15;
    myFpsPoint.y = 450;
    
    //insert fps text
    cv::putText(output_frame, sstmFps.str(), myFpsPoint, fontFace, fontScale, fontColor, fontThickness,cv::LINE_AA,false);
    time_last_cycle=end;
}

int main(){
    cv::VideoCapture capture(0);
    cv::Mat frame;
    cv::Mat output_frame;  
    char mode = 0; 
    std::string myOutString = "No filtering";
    start=clock();
    int counter=0;


    capture >> frame;
    unsigned char* grayImage = 
        (unsigned char*)malloc(sizeof(unsigned char)*frame.cols*frame.rows);

    unsigned char* d_ucharImage;
    cudaMalloc((void**)&d_ucharImage,
        sizeof(unsigned char)*frame.cols*frame.rows*frame.channels());
    unsigned char* d_grayImage;
    cudaMalloc((void**)&d_grayImage,
        sizeof(unsigned char)*frame.cols*frame.rows);
    unsigned int* d_histogram;
    cudaMalloc((void**)&d_histogram, sizeof(unsigned int)*HISTOGRAM_LENGTH);
    float *d_cdf;
    cudaMalloc((void**)&d_cdf, sizeof(unsigned int)*HISTOGRAM_LENGTH);



    for (;;){
        capture >> frame;
        if (frame.empty())
            break;
        imshow("Input", frame);
        output_frame = frame.clone();
        char key = cv::waitKey(30); //delay N millis, usually long enough to display and capture input
        if(key!=-1){
            mode = key;
            counter=0;
            start=clock();
        }
        switch (mode) {
        	case 'a': 
        	    histogramEqualizationOnHost(output_frame, frame, grayImage);   
        	    myOutString = "CPU Histogram Equalization";   	          
        	    break;
        	case 'b':
        	    histogramEqualizationOnDevice(output_frame, frame, d_ucharImage, d_grayImage, d_histogram, d_cdf);
        	    myOutString = "CUDA Histogram Equalization"; 
        	    break;     
        	case 'q':
        	case 'Q':
        	case 27:
        		output_frame.release();
	            frame.release();
                return 0;
        	default: 
                myOutString = "No filtering";
        	    break;
        }
        compute_fps(++counter, myOutString, output_frame);
        imshow("Output", output_frame);

    }
    output_frame.release();
	frame.release();

    free(grayImage);

    cudaFree(d_ucharImage);
    cudaFree(d_grayImage);
    cudaFree(d_cdf);
    cudaFree(d_histogram);    

    cudaDeviceReset();
	return 0;
}


void histogramEqualizationOnHost(cv::Mat& outputImage, cv::Mat& inputImage, 
    unsigned char* grayImage){

    int imageWidth;
    int imageHeight;
    int imageChannels;

    imageWidth = inputImage.cols;
    imageHeight = inputImage.rows;
    imageChannels = inputImage.channels();

	//#1
	h_RGBtoGS(grayImage, inputImage.data, imageHeight, imageWidth);

	//#2
	unsigned int* histogram;
	histogram=(unsigned int*)malloc(sizeof(unsigned int)*HISTOGRAM_LENGTH);
	memset(histogram,0,sizeof(unsigned int)*HISTOGRAM_LENGTH);
	h_Histogram(histogram, grayImage, imageWidth, imageHeight);

	//#3
	float *cdf;
	cdf=(float*)malloc(sizeof(float)*HISTOGRAM_LENGTH);
	h_Cdf(cdf, histogram, imageWidth, imageHeight);

    //#4
	float cdfmin;
	cdfmin = h_MinOfCdf(cdf);

	//#5
	h_Hef(outputImage.data, cdf, cdfmin, imageWidth, imageHeight, imageChannels);

	free(cdf);
	free(histogram);

}

void histogramEqualizationOnDevice(cv::Mat& outputImage, cv::Mat& inputImage, 
    unsigned char* d_ucharImage, unsigned char* d_grayImage,  unsigned int* d_histogram, float *d_cdf){

    int imageWidth;
    int imageHeight;
    int imageChannels;
    imageWidth = inputImage.cols;
    imageHeight = inputImage.rows;
    imageChannels = inputImage.channels();

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
   
    cudaMemcpyAsync(d_ucharImage, inputImage.data,
        sizeof(unsigned char)*imageWidth*imageHeight*imageChannels, cudaMemcpyHostToDevice, stream1);

    cudaMemsetAsync(d_histogram,0,sizeof(unsigned int)*HISTOGRAM_LENGTH, stream2);


    float cdfmin;
    float *d_cdfmin;
    cudaMalloc((void**)&d_cdfmin, sizeof(float));    


   	//#1
   	dim3 dimB(BSIZE2D, BSIZE2D);
   	dim3 dimG(imageWidth/BSIZE2D + 1, imageHeight / BSIZE2D + 1);
   	d_RGBtoGS<<<dimG,dimB, 0, stream1>>>(d_grayImage, d_ucharImage, imageHeight, imageWidth);



    //#2
    int numVals = imageWidth * imageHeight;
    dim3 dimBlockHisto(BSIZE,1,1);
    dim3 dimGridHisto(numVals / dimBlockHisto.x +1, 1, 1);
    d_Histogram<<<dimGridHisto, dimBlockHisto,  0, stream1>>>(d_histogram, d_grayImage, imageWidth, imageHeight);
 
    //#3
    d_Cdf<<<1,BSIZE,  0, stream1>>>(d_cdf, d_histogram, imageWidth, imageHeight, HISTOGRAM_LENGTH);

    //#4
    d_MinOfCdf<<<1,BSIZE,  0, stream1>>>(d_cdf, d_cdfmin, HISTOGRAM_LENGTH);

    cudaMemcpy(&cdfmin, d_cdfmin, sizeof(float), cudaMemcpyDeviceToHost);

    dim3 dimBlock(BSIZE2D, BSIZE2D, 1);
    dim3 dimGrid(imageWidth / dimBlock.x + 1,  imageHeight / dimBlock.y + 1, 1);

    //#4
    d_Hef<<< dimGrid, dimBlock,  0, stream1 >>>(d_ucharImage, d_cdf, cdfmin, imageWidth, imageHeight, imageChannels);


    cudaMemcpyAsync(outputImage.data, d_ucharImage,
    			sizeof(unsigned char)*imageWidth*imageHeight*imageChannels,
    			cudaMemcpyDeviceToHost, stream1);

    cudaFree(d_cdfmin);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

}
