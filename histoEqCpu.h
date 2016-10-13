/*
 * histoEqCpu.h
 *
 *  Created on: Oct 1, 2016
 *      Author: x
 */

#ifndef HISTOEQCPU_H_
#define HISTOEQCPU_H_



float h_p(unsigned int x, int width, int height){
    return ((float)x / (float)(width * height));
}

//Use the same clamp function you used in the Image Convolution MP
float h_clamp (float x, float start, float end){
    return std::min(std::max(x, start), end);
}


//#1
//Convert the image from RGB to GrayScale
//Implement a kernel that converts the the RGB image to GrayScale
void h_RGBtoGS(unsigned char* grayImage, unsigned char* ucharImage, int height, int width)
{
    for( int ii=0; ii<height; ii++)
        for( int jj=0; jj<width; jj++){
            int idx = ii * width + jj;
            //here channels is 3
            unsigned char r = ucharImage[3*idx];
            unsigned char g = ucharImage[3*idx + 1];
            unsigned char b = ucharImage[3*idx + 2];
            grayImage[idx] = (unsigned char) (0.21*r + 0.71*g + 0.07*b);
		}
}

//#2
//Compute the histogram of  grayImage
//Implement a kernel that computes the histogram (like in the lectures) of the image.
void h_Histogram(unsigned int *histogram, unsigned char *grayImage, int width, int height){

	for( int ii=0; ii < width * height; ii++)
        histogram[grayImage[ii]]++;
}

//#3
//Compute the Comulative Distribution Function of  histogram
//This is a scan operation like you have done in the previous lab
void h_Cdf(float* cdf, unsigned int* histogram, int width, int height){
    cdf[0] = h_p(histogram[0], width, height);
    for( int ii=1; ii<256; ii++){
		cdf[ii] = cdf[ii - 1] + h_p(histogram[ii], width, height);
	}
}

//#4
//Compute the minimum value of the CDF
//This is a reduction operation using the min function
float h_MinOfCdf(float* cdf){
    float cdfmin = cdf[0];
    for( int ii=1;ii<256;ii++)
        cdfmin = std::min(cdfmin, cdf[ii]);
	return cdfmin;
}

//#5
//Define the histogram equalization function
//The histogram equalization function ( correct ) remaps the cdf of the histogram
//of the image to a linear function
//and
//#6
//Apply the histogram equalization function
//Once you have implemented all of the above, then you are ready to correct the input image
void h_Hef(unsigned char* ucharImage, float *cdf, float cdfmin, int width, int height, int channels){
    for( int ii=0; ii<(width * height * channels);ii++)
        ucharImage[ii] = h_clamp(255*(cdf[ucharImage[ii]] - cdfmin)/(1.0 - cdfmin), 0, 255);
}




#endif /* HISTOEQCPU_H_ */
