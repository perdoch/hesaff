/*
 * Copyright (C) 2008-12 Michal Perdoch
 * All rights reserved.
 *
 * This file is part of the HessianAffine detector and is made available under
 * the terms of the BSD license (see the COPYING file).
 *
 */

/*
Detects hessian keypoint locations (but not affine shape) in a Gaussian pyramid. 
*/

#include <vector>
#include <string.h>
#include <algorithm>
#include "pyramid.h"
#include "helpers.h"
//#define print(msg) std::cout << msg << std::endl;

#ifdef WIN32
#define isnan(a)\
    (a != a)
#endif

#include <iostream>
using namespace std;
/* find blob point type from Hessian matrix H,
   we know that:
   - if H is positive definite it is a DARK blob,
   - if H is negative definite it is a BRIGHT blob
   - det H is negative it is a SADDLE point
   */
int getHessianPointType(float *ptr, float value)
{
    if(value < 0)
    {
        return HessianDetector::HESSIAN_SADDLE;
    }
    else
    {
        // at this point we know that 2x2 determinant is positive
        // so only check the remaining 1x1 subdeterminant
        float Lxx = (ptr[-1] - 2 * ptr[0] + ptr[1]);
        if(Lxx < 0)
        {
            return HessianDetector::HESSIAN_DARK;
        }
        else
        {
            return HessianDetector::HESSIAN_BRIGHT;
        }
    }
}

bool isMax(float val, const Mat &pix, int row, int col)
{
    for(int r = row - 1; r <= row + 1; r++)
    {
        // POTENTIAL ISSUE:
        // this will return true in a homogenous region.
        // Should this return false? if any point (other than the middle BUT
        // only when considering this-cur) is equal to val as well?
        const float *row = pix.ptr<float>(r);
        for(int c = col - 1; c <= col + 1; c++)
            if(row[c] > val)
            {
                return false;
            }
    }
    return true;
}

bool isMin(float val, const Mat &pix, int row, int col)
{
    /*
     Checks to see if pixel is a minima in the 8 connected region
     */
    for(int r = row - 1; r <= row + 1; r++)
    {
        const float *row = pix.ptr<float>(r);
        for(int c = col - 1; c <= col + 1; c++)
            if(row[c] < val)
            {
                return false;
            }
    }
    return true;
}

Mat HessianDetector::hessianResponse(const Mat &inputImage, float norm)
{
    /*
     Does 3x3 convolution to produce responce map 
     
     Computes the scale normalized determanant of the hessian matrix for a
     given image image (which is a level of the Gaussian pyramid).
    
    Step 1.1: Called from: main
    void HessianDetector::detectPyramidKeypoints(const Mat &image)
    void HessianDetector::detectOctaveHessianKeypoints(const Mat &firstLevel, float pixelDistance, 
                                                Mat &nextOctaveFirstLevel)
    RETURNS output_image
    */
    const int rows = inputImage.rows;
    const int cols = inputImage.cols;
    const int stride = cols;

    // allocate output
    // 32bit float 1 channel
    Mat outputImage(rows, cols, CV_32FC1);

    // setup input and output pointer to be centered at 1,0 and 1,1 resp.
    const float *in =  inputImage.ptr<float>(1);
    float      *out = outputImage.ptr<float>(1) + 1;


    /*
    Image:
      [(-1,-1)] [(-1 ,0)] [(-1, 1)] [(-1, 2)] 

      [( 0,-1)] [( 0, 0)] [( 0, 1)] [( 0, 2)] 
              
      [( 1,-1)] [( 1, 0)] [( 1, 1)] [( 1, 2)] 
             
      [( 2,-1)] [( 2, 0)] [( 2, 1)] [( 2, 2)]
             
      [( 3,-1)] [( 3, 0)] [( 3, 1)] [( 3, 2)]

    -----
      [(-1,-1)] [(-1 ,0)] [(-1, 1)] [(-1, 2)] 
               +-----------------------------
      [( 0,-1)]|[( 0, 0)] [( 0, 1)] [( 0, 2)] 
               |
      [( 1,-1)]|[    *in] [   *out] [( 1, 2)] 
               |
      [( 2,-1)]|[( 2, 0)] [( 2, 1)] [( 2, 2)]
               |
      [( 3,-1)]|[( 3, 0)] [( 3, 1)] [( 3, 2)]

    -----
      [(-1,-1)] [(-1 ,0)] [(-1, 1)] [(-1, 2)] 
               +-----------------------------
      [( 0,-1)]|[( 0, 0)] [( 0, 1)] [( 0, 2)] 
               |
      [    v11]|[    v21] [    v31] [( 1, 2)] 
               |
      [    v12]|[    v22] [    v32] [( 2, 2)]
               |
      [    v13]|[    v23] [    v33] [( 3, 2)]

    -----
      [(-1,-1)] [(-1 ,0)] [(-1, 1)] [(-1, 2)] 
               +-----------------------------
      [( 0,-1)]|[( 0, 0)] [( 0, 1)] [( 0, 2)] 
               |
      [    v11]|[    v21] [    v31] [( 1, 2)] 
               |
      [    v12]|[    v22] [    v32] [( 2, 2)]
               |
      [    v13]|[    v23] [    v33] [( 3, 2)]


      ----------
      Lxx and Lyy might be inverted 

      Lxx - turns out this does compute a 2nd derivative
      
        [v11] [v21] [v31]
              
        [v12] [v22] [v32]    1  -2  1
                               
        [v13] [v23] [v33]

      ----------
      Lyy
      
        [v11] [v21] [v31]      1
              
        [v12] [v22] [v32]     -2
                               
        [v13] [v23] [v33]      1

      ----------
      Lxy
      
        [v11] [v21] [v31]   -1/4        1/4
              
        [v12] [v22] [v32]         
                               
        [v13] [v23] [v33]    1/4       -1/4
     
     */

    float norm2 = norm * norm;

    /* move 3x3 window and convolve */
    for(int r = 1; r < rows - 1; ++r)
    {
        float v11, v12, v21, v22, v31, v32;
        /* fill in shift registers at the beginning of the row */
        // seems to perform wraparound at left and right edges
        v11 = in[-stride];   v12 = in[1 - stride];
        v21 = in[      0];   v22 = in[1         ];
        v31 = in[+stride];   v32 = in[1 + stride];
        /* move input pointer to (1,2) of the 3x3 square */
        in += 2;
        for(int c = 1; c < cols - 1; ++c)
        {
            /* fetch remaining values (last column) */
            const float v13 = in[-stride];
            const float v23 = *in;
            const float v33 = in[+stride];

            // compute 3x3 Hessian values from symmetric differences.
            // L(pt, scale) = [[Lxx, Lxy], [Lxy, Lyy]]
            // Lxx = (v21 - 2*v22 + v23) = ((v23 - v22) - (v22 - v21))  # 2nd derivative in x
            float Lxx = (v21 - (2 * v22) + v23);
            float Lyy = (v12 - (2 * v22) + v32);
            float Lxy = (v13 - v11 + v31 - v33) / 4.0f;

            // Compute the scale normalized hessian determanant and 
            // write to the ouptut image.
            // normalize and write out
            *out = (Lxx * Lyy - Lxy * Lxy) * norm2;

            /* move window */
            v11 = v12;
            v12 = v13;
            v21 = v22;
            v22 = v23;
            v31 = v32;
            v32 = v33;

            /* move input/output pointers */
            in++;
            out++;
        }
        out += 2;
    }
    return outputImage;
}

// it seems 0.6 works better than 0.5 (as in DL --- David Lowe --- paper)
// https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
#define MAX_SUBPIXEL_SHIFT 0.6

// we don't care about border effects
#define POINT_SAFETY_BORDER  3

void HessianDetector::localizeKeypoint(int r, int c, float curScale, float pixelDistance)
{
    /*
     
    - Localizes the keypoint in position and scale (but not shape) by fitting a
    parabola (2nd order taylor expansion), finding the extrema, relocalizing,
    and then iterating until convergence.  - Checks to make sure keypoint is
    not on an edge.

    
    Step 2:
    main
    0: void HessianDetector::detectPyramidKeypoints(const Mat &image)
    1: void HessianDetector::detectOctaveHessianKeypoints(const Mat &firstLevel, float pixelDistance, 
                                                   Mat &nextOctaveFirstLevel)
    1.2: void HessianDetector::findLevelKeypoints(float curScale, float pixelDistance)
    */
    const int cols = this->cur.cols;
    const int rows = this->cur.rows;

    float b[3] = {};
    float val = 0;
    bool converged = false;
    int nr = r, nc = c;

    for(int iter = 0; iter < 5; iter++)
    {
        // take current position
        r = nr;
        c = nc;

        float dxx = this->cur.at<float>(r, c - 1) - 2.0f * this->cur.at<float>(r, c) + this->cur.at<float>(r, c + 1);
        float dyy = this->cur.at<float>(r - 1, c) - 2.0f * this->cur.at<float>(r, c) + this->cur.at<float>(r + 1, c);
        float dss = this->low.at<float>(r, c) - 2.0f * this->cur.at<float>(r, c) + this->high.at<float>(r, c);

        float dxy = 0.25f * (
                this->cur.at<float>(r + 1, c + 1) - 
                this->cur.at<float>(r + 1, c - 1) - 
                this->cur.at<float>(r - 1, c + 1) + 
                this->cur.at<float>(r - 1, c - 1));
        // check edge like shape of the response function in first iteration
        if(0 == iter)
        {
            float edgeScore = (dxx + dyy) * (dxx + dyy) / (dxx * dyy - dxy * dxy);
            if(edgeScore >= this->edgeScoreThreshold || edgeScore < 0)
                // local neighbourhood looks like an edge
            {
                return;
            }
        }
        float dxs = 0.25f * (
                this->high.at<float>(r  , c + 1) - this->high.at<float>(r  , c - 1) - 
                 this->low.at<float>(r  , c + 1) +  this->low.at<float>(r  , c - 1));
        float dys = 0.25f * (
                this->high.at<float>(r + 1, c) - this->high.at<float>(r - 1, c) - 
                 this->low.at<float>(r + 1, c) +  this->low.at<float>(r - 1, c));

        float A[9];
        A[0] = dxx; A[1] = dxy; A[2] = dxs;
        A[3] = dxy; A[4] = dyy; A[5] = dys;
        A[6] = dxs; A[7] = dys; A[8] = dss;

        float dx = 0.5f * (this->cur.at<float>(r, c + 1) - this->cur.at<float>(r, c - 1));
        float dy = 0.5f * (this->cur.at<float>(r + 1, c) - this->cur.at<float>(r - 1, c));
        float ds = 0.5f * (this->high.at<float>(r, c)  - this->low.at<float>(r, c));

        b[0] = - dx;
        b[1] = - dy;
        b[2] = - ds;

        solveLinear3x3(A, b);
        // check if the solution is valid
        if(isnan(b[0]) || isnan(b[1]) || isnan(b[2]))
        {
            return;
        }
        // aproximate peak value
        val = this->cur.at<float>(r, c) + 0.5f * (dx * b[0] + dy * b[1] + ds * b[2]);
        // if we are off by more than MAX_SUBPIXEL_SHIFT, update the position and iterate again
        if(b[0] >  MAX_SUBPIXEL_SHIFT)
        {
            if(c < cols - POINT_SAFETY_BORDER)
            { nc++; }
            else
            { return; }
        }
        if(b[1] >  MAX_SUBPIXEL_SHIFT)
        {
            if(r < rows - POINT_SAFETY_BORDER)
            { nr++; }
            else
            { return; }
        }
        if(b[0] < -MAX_SUBPIXEL_SHIFT)
        {
            if(c > POINT_SAFETY_BORDER)
            { nc--; }
            else
            { return; }
        }
        if(b[1] < -MAX_SUBPIXEL_SHIFT)
        {
            if(r > POINT_SAFETY_BORDER)
            { nr--; }
            else
            { return; }
        }
        if(nr == r && nc == c)
        {
            // converged, displacement is sufficiently small, terminate here
            // TODO: decide if we want only converged local extrema...
            converged = true;
            break;
        }
    }
    // if spatial localization was all right and the scale is close enough...
    if(fabs(b[0]) > 1.5 || fabs(b[1]) > 1.5 || fabs(b[2]) > 1.5 || 
       fabs(val) < this->finalThreshold || this->octaveMap.at<unsigned char>(r, c) > 0)
    {
        return;
    }
    // mark we were here already
    this->octaveMap.at<unsigned char>(r, c) = 1;
    // output keypoint
    float scale = curScale * pow(2.0f, b[2] / par.numberOfScales);
    // set point type according to final location
    int type = getHessianPointType(blur.ptr<float>(r) + c, val);
    // point is now scale and translation invariant, add it...
    if(this->hessianKeypointCallback)
    {
        // Define subpixel and subscale location
        const float x = pixelDistance * (c + b[0]);
        const float y = pixelDistance * (r + b[1]);
        const float s = pixelDistance * scale;
        // Callback is connected to:
        // findAffineShape(blur, x, y, s, pixelDistance, type, response);
        // Call Step 3. in hessaff.cpp
        this->hessianKeypointCallback->onHessianKeypointDetected(
                this->prevBlur, x, y, s, pixelDistance, type, val); 
    }
}


void HessianDetector::findDenseLevelKeypoints(float curScale, float pixelDistance)
{
    // HACKED IN FUNCTION
    const int rows = this->octaveMap.rows;
    const int cols = this->octaveMap.cols;
    const float scale = curScale * pow(2.0f, 1.0f / par.numberOfScales);
    int type = -1;
    float val = 0;

    const int dense_stride = par.dense_stride;

    //std::cout << "here" << std::endl;
    for(int r = par.border; r < (rows - par.border); r+=dense_stride)
    {
        for(int c = par.border; c < (cols - par.border); c+=dense_stride)
        {
            // HACK: this is not a hessian keypoint, these are
            // determenistic computed grid keypoints, but we are going to use this
            // callback to hack in dense keypoint generation
            const float x = pixelDistance * c;
            const float y = pixelDistance * r;
            const float s = pixelDistance * scale;
            // Callback is connected to:
            // findAffineShape(blur, x, y, s, pixelDistance, type, response);
            this->hessianKeypointCallback->onHessianKeypointDetected(
                    this->prevBlur, x, y, s, pixelDistance, type, val); // Call Step 3. in hessaff.cpp
        }
    }
}

void HessianDetector::detectOctaveDenseKeypoints(const Mat &firstLevel, float pixelDistance, Mat &nextOctaveFirstLevel)
{
    //CV_8UC1 means an 8-bit unsigned single-channel matrix
    this->octaveMap = Mat::zeros(firstLevel.rows, firstLevel.cols, CV_8UC1);
    float sigmaStep = pow(2.0f, 1.0f / (float) par.numberOfScales);
    float curSigma = par.initialSigma;
    this->blur = firstLevel;
    int numLevels = 1;

    for(int i = 1; i < par.numberOfScales + 2; i++)
    {
        // compute the increase necessary for the next level and compute the next level
        float sigma = curSigma * sqrt(sigmaStep * sigmaStep - 1.0f);
        Mat nextBlur = gaussianBlur(this->blur, sigma); //Helper function
        sigma = curSigma * sigmaStep; // the next level sigma
        numLevels++;
        // if we have three consecutive responses
        if(numLevels == 3)
        {
            // find keypoints in this part of octave for curLevel
            this->findDenseLevelKeypoints(curSigma, pixelDistance); //Call Step 1.2
            numLevels--;
        }
        if(i == par.numberOfScales)
        {
            // downsample the right level for the next octave
            nextOctaveFirstLevel = halfImage(nextBlur);    // Helper Function
        }
        this->prevBlur = this->blur;
        this->blur = nextBlur;
        // shift to the next response
        this->low = this->cur;
        this->cur = this->high;
        curSigma *= sigmaStep;
    }
}

void HessianDetector::findLevelKeypoints(float curScale, float pixelDistance)
{
    /*
    Step 1.2: Called from main

    Finds extreme points in space and scale

    0: void HessianDetector::detectPyramidKeypoints(const Mat &image)
    1: void HessianDetector::detectOctaveHessianKeypoints(const Mat &firstLevel, float pixelDistance,
                                                    Mat &nextOctaveFirstLevel)
    */
    assert(par.border >= 2);
    const int rows = this->cur.rows;
    const int cols = this->cur.cols;
    for(int r = par.border; r < (rows - par.border); r++)
    {
        for(int c = par.border; c < (cols - par.border); c++)
        {
            const float val = this->cur.at<float>(r, c);
            //If current val is an extreme point in (x,y,sigma)
            // either positive -> local max. or negative -> local min.
            const bool pass_pos_thresh = (val > this->positiveThreshold &&
                (isMax(val, this->cur, r, c) && 
                 isMax(val, this->low, r, c) && 
                 isMax(val, this->high, r, c))
                );
            const bool pass_neg_thresh = (val < this->negativeThreshold && 
                (isMin(val, this->cur, r, c) && 
                 isMin(val, this->low, r, c) && 
                 isMin(val, this->high, r, c))
                );
            if(pass_pos_thresh || pass_neg_thresh)
            {
                // Localize extreme point to subpixel and subscale accuracy
                this->localizeKeypoint(r, c, curScale, pixelDistance);    // Call Step 2
            }
        }
    }
}

//Step1: Called from:
// main
// void HessianDetector::detectPyramidKeypoints(const Mat &image)
void HessianDetector::detectOctaveHessianKeypoints(const Mat &firstLevel, float pixelDistance, Mat &nextOctaveFirstLevel)
{
    //CV_8UC1 means an 8-bit unsigned single-channel matrix
    this->octaveMap = Mat::zeros(firstLevel.rows, firstLevel.cols, CV_8UC1);
    float sigmaStep = pow(2.0f, 1.0f / (float) par.numberOfScales);
    float curSigma = par.initialSigma;
    this->blur = firstLevel;
    // Calculate hessian responce at the octave's base level
    this->cur = hessianResponse(this->blur, curSigma * curSigma);
    int numLevels = 1;

    for(int i = 1; i < par.numberOfScales + 2; i++)
    {
        // compute the increase necessary for the next level and compute the next level
        float sigma = curSigma * sqrt(sigmaStep * sigmaStep - 1.0f);
        Mat nextBlur = gaussianBlur(this->blur, sigma); //Helper function
        sigma = curSigma * sigmaStep; // the next level sigma
        // compute hessian response for current (scale) level
        this->high = this->hessianResponse(nextBlur, sigma * sigma); //Call Step 1.1
        numLevels++;
        // if we have three consecutive responses
        if(numLevels == 3)
        {
            // find keypoints in this part of octave for curLevel
            this->findLevelKeypoints(curSigma, pixelDistance); //Call Step 1.2
            numLevels--;
        }
        if(i == par.numberOfScales)
        {
            // downsample the right level for the next octave
            nextOctaveFirstLevel = halfImage(nextBlur);    // Helper Function
        }
        this->prevBlur = this->blur;
        this->blur = nextBlur;
        // shift to the next response
        this->low = this->cur;
        this->cur = this->high;
        curSigma *= sigmaStep;
    }
}

// Entry point of image. Step 0
// Called from main
void HessianDetector::detectPyramidKeypoints(const Mat &image)
{
    float curSigma = 0.5f; //Not sure why this starts at .5 and not 0
    float pixelDistance = 1.0f; //Pixels step size
    Mat   firstLevel;
    // Copy the image into the first level
    firstLevel = image.clone();
    // prepare first octave input image
    // Given the param initialSigma, make
    // sure we are on that level of scalespace
    if(par.initialSigma > curSigma)
    {
        //Calculate sigma to get to initial sigma
        float sigma = sqrt(par.initialSigma * par.initialSigma - curSigma * curSigma);
        gaussianBlurInplace(firstLevel, sigma);
    }
    // detect keypoints at scales of increasing sigma
    // until there is the image is too small.
    // This keypoint detection happens at different scales
    // It calls the octave keypoint detector which detects keypoints
    // between scales, and blurs the image in place. when sigma doubles
    // we downsample the image in this loop and procede to check the octaves in that scale
    int minSize = 2 * par.border + 2;
    int num_blurs = 0;
    int current_pyramid_level = 0;
    const bool use_dense = par.use_dense;
    while(firstLevel.rows > minSize && firstLevel.cols > minSize)
    {
        Mat nextOctaveFirstLevel; //Outvar
        if (use_dense)
        {
            // # hacked in step
            detectOctaveDenseKeypoints(firstLevel, pixelDistance, nextOctaveFirstLevel);
        }
        else
        {
            detectOctaveHessianKeypoints(firstLevel, pixelDistance, nextOctaveFirstLevel); //Call Step 1
        }
        pixelDistance *= 2.0; // Effectively increase sigma by varying the pixel step size
        // Overwrite firstlevel with the next level blur
        firstLevel = nextOctaveFirstLevel; // Overwrite firstLevel in place
        //cout << "num blurs: " << ++num_blurs << std::endl;
        current_pyramid_level++;
        if (par.maxPyramidLevels != -1 && par.maxPyramidLevels <= current_pyramid_level)
        {
            break;
        }
    }
}
