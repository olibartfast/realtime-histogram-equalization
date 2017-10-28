# Real time histogram equalization

The purpose of this project is to implement an efficient histogramming equalization algorithm for an input frame acquired from a webcam device in real time, the image is represented as RGB unsigned char values, it will be converted to grayscale and then computed the histogram. Based on the histogram, will be obtained an equalization function then applied to the original image to get the color corrected image, basically fulfilling the following steps:

* Convert the image from RGB to grayscale
* Compute the histogram of the image
* Compute the scan and prefix sum of the histogram to arrive at the histogram equalization function
* Apply the equalization function to get the color corrected image

## Background details
For an image that represent a full color space is expected to have a uniform distribution of luminosity values, this means that if we compute the Cumulative Distribution Function (CDF) we expect a linear curve for a color equalized image, if not we expect the curve to be non-linear (Picture 3). The algorithm equalizes the curve by computing a transformation function to map the original CDF to the desired one, being an almost linear function (Picture 5).

![Alt Text](./images/input-image.jpg)
_**Picture 1: Input image**_



We first need to convert the image (Picture 1) to gray scale (Picture 2) by computing it’s luminosity values. These represent the brightness of the image and would allow us to simplify the histogram computation.



![Alt Text](./images/grayscale-image.jpg)
_**Picture 2: Grayscale image**_



![Alt Text](./images/non-linear-cdf.png)  
_**Picture 3: For an image that is not color equalized we expect the curve to be non-linear**_


![Alt Text](./images/equalized-image.jpg)  
_**Picture 4: The computed transformation is applied to the original image to produce the equalized image**_  


![Alt Text](./images/non-linear-cdf.png)  
_**Picture 5: the desired CDF being an almost linear function after histogram equalization**_


PDF version: https://goo.gl/mNM2Ky

Demo on YouTube using Jetson TK1: https://www.youtube.com/watch?v=Vje8XCtam7A

Demo on YouTube using an I7-3610QM @ 2.3 GHz, GPU Nvidia GeForce 640M: https://www.youtube.com/watch?v=LHSSxSk55hQ
