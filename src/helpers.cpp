/*
 * Copyright (C) 2008-12 Michal Perdoch
 * All rights reserved.
 *
 * This file is part of the HessianAffine detector and is made available under
 * the terms of the BSD license (see the COPYING file).
 *
 */

#include <cmath>
#include <iostream>
#include <opencv2/imgproc.hpp>

#include "helpers.h"

using namespace cv;
using namespace std;

//#ifndef WIN32
#if 0
#define _HAS_GET_TIME

#include <sys/times.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdio.h>

double getTime()
{
#ifdef _POSIX_CPUTIME
    struct timespec ts;
    if(!clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts))
    {
        return (double)(ts.tv_sec) + (double)(ts.tv_nsec) / 1.0e9;
    }
    else
#endif
    {
        // fall back to standard unix time
        struct timeval tv;
        gettimeofday(&tv, 0);
        return (double)(tv.tv_sec) + (double)(tv.tv_usec) / 1.0e6;
    }
}
#endif

#ifdef DEBUG_HELPERS
#undef DEBUG_HELPERS
#endif
#define DEBUG_HELPERS

#ifdef DEBUG_HELPERS
#define printDBG(msg) std::cerr << "[helpers.c] " << msg << std::endl;
#define write(msg) std::cerr << msg;
#else
#define printDBG(msg);
#endif


void run_system_command(std::string cmd_str)
{
    printDBG("+ ----- RUNNING COMMAND ----- ")
    printDBG(cmd_str);
    int retcode = system(cmd_str.c_str());
    printDBG(" retcode = " << retcode)
    printDBG("L _______ FINISHED RUNNING COMMAND _______ ")
    if (retcode != 0)
    {
        printDBG("FAILURE")
        exit(1);
    }
}


void make_2d_gauss_patch_01(int rows, int cols, float sigma0, float sigma1, cv::Mat& gauss_weights)
{
    double d0_max, d0_min, d1_max, d1_min;
    cv::Mat gauss_kernel_d0 = cv::getGaussianKernel(rows, sigma0, CV_32F);
    cv::Mat gauss_kernel_d1 = cv::getGaussianKernel(cols, sigma1, CV_32F);
    cv::minMaxLoc(gauss_kernel_d0, &d0_min, &d0_max);
    cv::minMaxLoc(gauss_kernel_d1, &d1_min, &d1_max);
    gauss_kernel_d0 = gauss_kernel_d0.mul(1.0f / d0_max);
    gauss_kernel_d1 = gauss_kernel_d1.mul(1.0f / d1_max);
    //cv::Mat gauss_weights = gauss_kernel_d1.dot(gauss_kernel_d0.t());
    gauss_weights = gauss_kernel_d1 * gauss_kernel_d0.t();
}


template <typename ValueType>
void swap(ValueType *a, ValueType *b)
{
    ValueType tmp = *a;
    *a = *b;
    *b = tmp;
}


void solveLinear3x3(float *A, float *b)
{
    // find pivot of first column
    int i = 0;
    float *pr = A;
    float vp = abs(A[0]);
    float tmp = abs(A[3]);
    if(tmp > vp)
    {
        // pivot is in 1st row
        pr = A + 3;
        i = 1;
        vp = tmp;
    }
    if(abs(A[6]) > vp)
    {
        // pivot is in 2nd row
        pr = A + 6;
        i = 2;
    }

    // swap pivot row with first row
    if(pr != A)
    {
        swap(pr, A);
        swap(pr + 1, A + 1);
        swap(pr + 2, A + 2);
        swap(b + i, b);
    }

    // fixup elements 3,4,5,b[1]
    vp = A[3] / A[0];
    A[4] -= vp * A[1];
    A[5] -= vp * A[2];
    b[1] -= vp * b[0];

    // fixup elements 6,7,8,b[2]]
    vp = A[6] / A[0];
    A[7] -= vp * A[1];
    A[8] -= vp * A[2];
    b[2] -= vp * b[0];

    // find pivot in second column
    if(abs(A[4]) < abs(A[7]))
    {
        swap(A + 7, A + 4);
        swap(A + 8, A + 5);
        swap(b + 2, b + 1);
    }

    // fixup elements 7,8,b[2]
    vp = A[7] / A[4];
    A[8] -= vp * A[5];
    b[2] -= vp * b[1];

    // solve b by back-substitution
    b[2] = (b[2]) / A[8];
    b[1] = (b[1] - A[5] * b[2]) / A[4];
    b[0] = (b[0] - A[2] * b[2] - A[1] * b[1]) / A[0];
}


void rotateAffineTransformation(float &a11, float &a12, float &a21, float &a22, float &theta)
{
    double a = static_cast<double>(a11), b = static_cast<double>(a12),
           c = static_cast<double>(a21), d = static_cast<double>(a22);
    double sin_ = sin(theta);
    double cos_ = cos(theta);
    a11 = (cos_ * a) + (-sin_ * c);
    a12 = (sin_ * a) + (cos_ * c);
    a21 = (cos_ * b) + (-sin_ * d);
    a22 = (sin_ * b) + (cos_ * d);
    //a11 = (a *  cos_) + (b * sin_);
    //a12 = (c *  cos_) + (d * sin_);
    //a21 = (a * -sin_) + (b * cos_);
    //a22 = (c * -sin_) + (d * cos_);
}


void rectifyAffineTransformationUpIsUp(float &a11, float &a12, float &a21, float &a22)
{
    // Rotates a matrix into its lower triangular form
    double a = static_cast<double>(a11), b = static_cast<double>(a12),
           c = static_cast<double>(a21), d = static_cast<double>(a22);
    double det = sqrt(abs(a * d - b * c));
    double b2a2 = sqrt(b * b + a * a);
    a11 = b2a2 / det;
    a12 = 0;
    a21 = (d * b + c * a) / (b2a2 * det);
    a22 = det / b2a2;
}


void rectifyAffineTransformationUpIsUp(float *U)
{
    rectifyAffineTransformationUpIsUp(U[0], U[1], U[2], U[3]);
}


void computeGaussMask(Mat &mask)
{
    // Input: blank matrix, populates matrix with values of a 2D gaussian
    // http://en.wikipedia.org/wiki/Gaussian_function
    // (has magic constants)
    int size = mask.cols;
    int halfSize = size >> 1;

    // fit 3*sigma into half_size
    float scale  = float(halfSize) / 3.0f;  // ~33.3% of radius is relevant
    float scale2 = -2.0f * scale * scale;  // negate and divide by two for gauss exponent

    // Compute gaussian equation in one dimension: populate tmp
    // the mean (mu) is 0
    // the standard devation (sigma) is scale2
    float *tmp = new float[halfSize + 1];
    for(int i = 0; i <= halfSize; i++)
    {
        tmp[i] = exp((float(i * i) / scale2));
    }
    // I'm not exactely sure what this is, but this adds a small bump to the tail
    // Gaussians are closed under convolution and multiplication, but probably not addition
    // so this must be some sort of hack
    // (maybe this is just to upweight the tails a little bit)
    int endSize = int(ceil(scale * 5.0f) - halfSize);
    for(int i = 1; i < endSize; i++)
    {
        tmp[halfSize - i] += exp((float((i + halfSize) * (i + halfSize)) / scale2));
    }

    // Gaussian mask is symmetric \in isometric, walks over the first quadrent only
    // and sets all 4 mirrored quadrents
    for(int i = 0; i <= halfSize; i++)
    {
        for(int j = 0; j <= halfSize; j++)
        {
            float gauss_weight = tmp[i] * tmp[j];
            mask.at<float>(i + halfSize, -j + halfSize) = gauss_weight;
            mask.at<float>(-i + halfSize,  j + halfSize) = gauss_weight;
            mask.at<float>(i + halfSize,  j + halfSize) = gauss_weight;
            mask.at<float>(-i + halfSize, -j + halfSize) = gauss_weight;
        }
    }
    delete [] tmp;
}


void computeCircularGaussMask(Mat &mask)
{
    int size = mask.cols;
    int halfSize = size >> 1;
    float r2 = float(halfSize * halfSize);  // radius squared
    float sigma2 = 0.9f * r2;  // ~95% of radius is relevant
    float disq;
    float *mp = mask.ptr<float>(0);
    for(int i = 0; i < mask.rows; i++)
    {
        for(int j = 0; j < mask.cols; j++)
        {
            // The mask is populated with Gaussian values
            // as the function of the radius
            disq = float((i - halfSize) * (i - halfSize) + (j - halfSize) * (j - halfSize));
            *mp++ = (disq < r2) ? exp(-disq / sigma2) : 0;
        }
    }
}


void invSqrt(float &a, float &b, float &c, float &eigval1, float &eigval2)
{
    // Inverse matrix square root
    // if Z = V.dot(V)
    // Given matrix Z
    // find matrix inv(V), which is the inverse square root of Z
    double t, r;
    if(b != 0)
    {
        r = double(c - a) / (2 * b);
        if(r >= 0)
        {
            t = 1.0 / (r +::sqrt(1 + r * r));
        }
        else
        {
            t = -1.0 / (-r +::sqrt(1 + r * r));
        }
        r = 1.0 /::sqrt(1 + t * t); /* c */
        t = t * r;             /* s */
    }
    else
    {
        r = 1;
        t = 0;
    }
    double x, z, d;

    x = 1.0 / sqrt(r * r * a - 2 * r * t * b + t * t * c);
    z = 1.0 / sqrt(t * t * a + 2 * r * t * b + r * r * c);

    d = sqrt(x * z);
    x /= d;
    z /= d;
    // let eigval1 be the greater eigenvalue
    if(x < z)
    {
        eigval1 = float(z);
        eigval2 = float(x);
    }
    else
    {
        eigval1 = float(x);
        eigval2 = float(z);
    }
    // output square root
    a = float(r * r * x + t * t * z);
    b = float(-r * t * x + t * r * z);
    c = float(t * t * x + r * r * z);
}


bool getEigenvalues(float a, float b, float c, float d, float &eigval1, float &eigval2)
{
    float trace = a + d;
    float delta1 = (trace * trace - 4 * (a * d - b * c));
    if(delta1 < 0)
    {
        return false;
    }
    float delta = sqrt(delta1);

    eigval1 = (trace + delta) / 2.0f;
    eigval2 = (trace - delta) / 2.0f;
    return true;
}


// check if we are not too close to boundary of the image/
bool interpolateCheckBorders(const Mat &im, float ofsx, float ofsy,
                             float a11, float a12, float a21, float a22, const Mat &res)
{
    /*
     Mirrors checks in interpolateCheckBorders but
     Simply returns true or false, does not affect state
     */
    // does not do interpolation, just checks if we can
    const int width  = im.cols - 2;
    const int height = im.rows - 2;
    const int halfWidth  = res.cols >> 1;
    const int halfHeight = res.rows >> 1;
    float x[4];
    x[0] = static_cast< float >(-halfWidth);
    x[1] = static_cast< float >(-halfWidth);
    x[2] = static_cast< float >(+halfWidth);
    x[3] = static_cast< float >(+halfWidth);
    float y[4];
    y[0] = static_cast< float >(-halfHeight);
    y[1] = static_cast< float >(+halfHeight);
    y[2] = static_cast< float >(-halfHeight);
    y[3] = static_cast< float >(+halfHeight);
    for(int i = 0; i < 4; i++)
    {
        float imx = ofsx + x[i] * a11 + y[i] * a12;
        float imy = ofsy + x[i] * a21 + y[i] * a22;
        if(floor(imx) <= 0 || floor(imy) <= 0 || ceil(imx) >= width || ceil(imy) >= height)
        {
            return true;
        }
    }
    return false;
}


bool interpolate(const Mat &im, float ofsx, float ofsy,
                 float a11, float a12, float a21, float a22, Mat &res)
{
    /*
    extracts a patch from im, corresponding to the keypoint using bilinear interpolation

    Args:
        im - image to extract patch from
        (ofsx, ofsy, a11, a12, a21, a22) - ellipse paramaters of patch to extract

    OutVar:
         res - patch of image im found using using elliptical shape

    Returns:
        bool: True if the patch was on the image boundary

    */
    bool ret = false;
    // input size (-1 for the safe bilinear interpolation)
    const int width = im.cols - 1;
    const int height = im.rows - 1;
    // output size
    const int halfWidth  = res.cols >> 1;
    const int halfHeight = res.rows >> 1;
    float *out = res.ptr<float>(0);
    for(int j = -halfHeight; j <= halfHeight; ++j)
    {
        const float rx = ofsx + j * a12;
        const float ry = ofsy + j * a22;
        for(int i = -halfWidth; i <= halfWidth; ++i)
        {
            float wx = rx + i * a11;
            float wy = ry + i * a21;
            const int x = (int) floor(wx);
            const int y = (int) floor(wy);
            if(x >= 0 && y >= 0 && x < width && y < height)
            {
                // compute weights
                wx -= x;
                wy -= y;
                // bilinear interpolation
                *out++ =
                    (1.0f - wy) * ((1.0f - wx) * im.at<float>(y, x)   + wx * im.at<float>(y, x + 1)) +
                    (wy) * ((1.0f - wx) * im.at<float>(y + 1, x) + wx * im.at<float>(y + 1, x + 1));
            }
            else
            {
                *out++ = 0;
                ret =  true; // touching boundary of the input
            }
        }
    }
    return ret;
}


void photometricallyNormalize(Mat &image, const Mat &binaryMask,
                              float &mean, float &var)
{
    // magic? Attempts to normalize extreme lighting conditions
    const int width = image.cols;
    const int height = image.rows;
    float sum = 0;  // sum of nonmasked pixels
    float gsum = 0; // number of nonmasked pixels
    // Compute average of nonmasked pixels
    for(int j = 0; j < height; j++)
        for(int i = 0; i < width; i++)
        {
            if(binaryMask.at<float>(j, i) > 0)
            {
                sum += image.at<float>(j, i);
                gsum ++;
            }
        }


    mean = sum / gsum;
    // Compute variance of nonmasked pixels
    var = 0;
    for(int j = 0; j < height; j++)
        for(int i = 0; i < width; i++)
        {
            if(binaryMask.at<float>(j, i) > 0)
            {
                var += (mean - image.at<float>(j, i)) * (mean - image.at<float>(j, i));
            }
        }

    var = ::sqrt(var / gsum);
    // if variance is too low, don't do anything
    if(var < 0.0001)
    {
        return;
    }
    // Normalize the mean to be 128, and repopulate the patch
    // keeping the same relative variations
    float fac = 50.0f / var;
    for(int j = 0; j < height; j++)
        for(int i = 0; i < width; i++)
        {
            image.at<float>(j, i) = 128 + fac * (image.at<float>(j, i) - mean);
            // Clamp to byte range
            if(image.at<float>(j, i) > 255)
            {
                image.at<float>(j, i) = 255;
            }
            if(image.at<float>(j, i) < 0)
            {
                image.at<float>(j, i) = 0;
            }
        }
}


Mat gaussianBlur(const Mat input, float sigma)
{
    Mat ret(input.rows, input.cols, input.type());
    int size = (int)(2.0 * 3.0 * sigma + 1.0);
    if(size % 2 == 0)
    {
        size++;
    }
    GaussianBlur(input, ret, Size(size, size), sigma, sigma, BORDER_REPLICATE); //opencv cv::GaussianBlur
    return ret;
}


void gaussianBlurInplace(Mat &inplace, float sigma)
{
    int size = (int)(2.0 * 3.0 * sigma + 1.0);
    if(size % 2 == 0)
    {
        size++;
    }
    GaussianBlur(inplace, inplace, Size(size, size), sigma, sigma, BORDER_REPLICATE);
}


Mat doubleImage(const Mat &input)
{
    Mat n(input.rows * 2, input.cols * 2, input.type());
    const float *in = input.ptr<float>(0);

    for(int r = 0; r < input.rows - 1; r++)
        for(int c = 0; c < input.cols - 1; c++)
        {
            const int r2 = r << 1;
            const int c2 = c << 1;
            n.at<float>(r2, c2)     = in[0];
            n.at<float>(r2 + 1, c2)   = 0.5f * (in[0] + in[input.step]);
            n.at<float>(r2, c2 + 1)   = 0.5f * (in[0] + in[1]);
            n.at<float>(r2 + 1, c2 + 1) = 0.25f * (in[0] + in[1] + in[input.step] + in[input.step + 1]);
            ++in;
        }
    for(int r = 0; r < input.rows - 1; r++)
    {
        const int r2 = r << 1;
        const int c2 = (input.cols - 1) << 1;
        n.at<float>(r2, c2)   = input.at<float>(r, input.cols - 1);
        n.at<float>(r2 + 1, c2) = 0.5f * (input.at<float>(r, input.cols - 1) + input.at<float>(r + 1, input.cols - 1));
    }
    for(int c = 0; c < input.cols - 1; c++)
    {
        const int r2 = (input.rows - 1) << 1;
        const int c2 = c << 1;
        n.at<float>(r2, c2)   = input.at<float>(input.rows - 1, c);
        n.at<float>(r2, c2 + 1) = 0.5f * (input.at<float>(input.rows - 1, c) + input.at<float>(input.rows - 1, c + 1));
    }
    n.at<float>(n.rows - 1, n.cols - 1) = n.at<float>(input.rows - 1, input.cols - 1);
    return n;
}


Mat halfImage(const Mat &input)
{
    Mat n(input.rows / 2, input.cols / 2, input.type());
    float *out = n.ptr<float>(0);
    for(int r = 0, ri = 0; r < n.rows; r++, ri += 2)
        for(int c = 0, ci = 0; c < n.cols; c++, ci += 2)
        {
            *out++ = input.at<float>(ri, ci);
        }
    return n;
}


bool almost_eq(float f1, float f2)
{
    float thresh = static_cast< float >(1E-10);
    return fabs(f1 - f2) < thresh;
}

void computeGradient(const cv::Mat &img, cv::Mat &gradx, cv::Mat &grady)
{
    const int width = img.cols;
    const int height = img.rows;
    // For each pixel in the image
    for(int r = 0; r < height; ++r)
        for(int c = 0; c < width; ++c)
        {
            float xgrad, ygrad;
            if(c == 0)
            {
                xgrad = img.at<float>(r, c + 1) - img.at<float>(r, c);
            }
            else if(c == width - 1)
            {
                xgrad = img.at<float>(r, c) - img.at<float>(r, c - 1);
            }
            else
            {
                xgrad = img.at<float>(r, c + 1) - img.at<float>(r, c - 1);
            }

            if(r == 0)
            {
                ygrad = img.at<float>(r + 1, c) - img.at<float>(r, c);
            }
            else if(r == height - 1)
            {
                ygrad = img.at<float>(r, c) - img.at<float>(r - 1, c);
            }
            else
            {
                ygrad = img.at<float>(r + 1, c) - img.at<float>(r - 1, c);
            }

            gradx.at<float>(r, c) = xgrad;
            grady.at<float>(r, c) = ygrad;
        }
}

/*void htool::makeCvHistFromHistogram(Histogram<float>& hist, CvHistogram& cvHist)
{
    int size = hist.data.size();
    float* data = &(hist.data[0]);
    float* rangeArr = &(hist.edges[0]);
    cvMakeHistHeaderForArray(1,&size,&cvHist,data,&rangeArr,0);
}*/
