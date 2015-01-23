/*
 * Copyright (C) 2008-12 Michal Perdoch
 * All rights reserved.
 *
 * This file is part of the HessianAffine detector and is made available under
 * the terms of the BSD license (see the COPYING file).
 *
 */

#include "affine.h"

// Gravity points downward = tau / 4 = pi / 2
#ifndef M_GRAVITY_THETA
#define M_GRAVITY_THETA 1.570795
// relative to gravity
#define R_GRAVITY_THETA 0
#endif

#ifdef MYDEBUG
#undef MYDEBUG
#endif
//#define MYDEBUG

#ifdef MYDEBUG
#define printDBG(msg) std::cout << "[hesaff.c] " << msg << std::endl;
#define write(msg) std::cout << msg;
#else
#define printDBG(msg);
#endif

using namespace cv;

// Step 3:
// main
//   0: void HessianDetector::detectPyramidKeypoints(const Mat &image)
//   1: void HessianDetector::detectOctaveKeypoints(const Mat &firstLevel, float pixelDistance, Mat &nextOctaveFirstLevel)
// 1.2: void HessianDetector::findLevelKeypoints(float curScale, float pixelDistance)
//   2: void HessianDetector::localizeKeypoint(int r, int c, float curScale, float pixelDistance)
bool AffineShape::findAffineShape(const Mat &blur, float x, float y, float s, float pixelDistance, int type, float response)
{
    float eigen_ratio_act = 0.0f, eigen_ratio_bef = 0.0f;
    float u11 = 1.0f, u12 = 0.0f, u21 = 0.0f, u22 = 1.0f, l1 = 1.0f, l2 = 1.0f;
    float lx = x / pixelDistance, ly = y / pixelDistance;
    float ratio = s / (par.initialSigma * pixelDistance);
    // kernel size...
    const int maskPixels = par.smmWindowSize * par.smmWindowSize;

    for(int l = 0; l < par.maxIterations; l ++)
    {
        // warp input according to current shape matrix
        interpolate(blur, lx, ly, u11 * ratio, u12 * ratio, u21 * ratio, u22 * ratio, img); //Helper function

        // compute SMM on the warped patch
        float a = 0, b = 0, c = 0;
        float *maskptr = mask.ptr<float>(0);
        float *pfx = fx.ptr<float>(0), *pfy = fy.ptr<float>(0);

        computeGradient(img, fx, fy); // Defined in this file

        // estimate SMM (second moment matrix)
        for(int i = 0; i < maskPixels; ++i)
        {
            const float v = (*maskptr);
            const float gxx = *pfx;
            const float gyy = *pfy;
            const float gxy = gxx * gyy;

            a += gxx * gxx * v;
            b += gxy * v;
            c += gyy * gyy * v;
            pfx++;
            pfy++;
            maskptr++;
        }
        a /= maskPixels;
        b /= maskPixels;
        c /= maskPixels;

        // compute inverse sqrt of the SMM
        invSqrt(a, b, c, l1, l2);

        // update eigen ratios
        eigen_ratio_bef = eigen_ratio_act;
        eigen_ratio_act = 1 - l2 / l1;

        // accumulate the affine shape matrix
        float u11t = u11, u12t = u12;

        u11 = a * u11t + b * u21;
        u12 = a * u12t + b * u22;
        u21 = b * u11t + c * u21;
        u22 = b * u12t + c * u22;

        // compute the eigen values of the shape matrix
        if(!getEigenvalues(u11, u12, u21, u22, l1, l2))
        {
            break;
        }

        // leave on too high anisotropy
        if((l1 / l2 > 6) || (l2 / l1 > 6))
        {
            break;
        }

        if(eigen_ratio_act < par.convergenceThreshold && eigen_ratio_bef < par.convergenceThreshold)
        {
            if(affineShapeCallback)
            {
                affineShapeCallback->onAffineShapeFound(blur, x, y, s, pixelDistance, u11, u12, u21, u22, type, response, l); // Call Step 4
            }
            return true;
        }
    }
    return false;
}


//Called by hessaff.cpp
bool AffineShape::normalizeAffine(const Mat &img,
                                  float x, float y,
                                  float s,
                                  float a11, float a12,
                                  float a21, float a22,
                                  float ori)
{
    // img is passed from onAffineShapeFound as this->image
    if(!almost_eq(ori, R_GRAVITY_THETA))
    {
        // rotate relative to the gravity vector
        float ori_offst = (ori - R_GRAVITY_THETA);
        printDBG("Rotating Patch ori=" << ori << "; offst_ori=" << ori_offst)
        rotateAffineTransformation(a11, a12, a21, a22, ori_offst); // helper
    }

    // determinant == 1 assumed (i.e. isotropic scaling should be separated in mrScale
    assert(fabs(a11 * a22 - a12 * a21 - 1.0f) < 0.01);
    //    mrSize = 3.0f*sqrt(3.0f);
    // half patch size in pixels of image
    float mrScale = ceil(s * par.mrSize);
    // enforce size to be odd
    int   patchImageSize = 2 * int(mrScale) + 1;
    // patchSize = 41;
    // patch size in image / patch size -> amount of down/up sampling
    float imageToPatchScale = float(patchImageSize) / float(par.patchSize);
    // is patch touching boundary? if yes, ignore this feature
    // does not affect state
    if(interpolateCheckBorders(img, x, y, a11 * imageToPatchScale,
                               a12 * imageToPatchScale, a21 * imageToPatchScale,
                               a22 * imageToPatchScale, this->patch))
    {
        return true;
    }

    if(imageToPatchScale > 0.4)
    {
        // CASE 1: Bigger patches (in image space)
        // the pixels in the image are 0.4 apart + the affine deformation
        // leave +1 border for the bilinear interpolation
        patchImageSize += 2;
        size_t wss = patchImageSize * patchImageSize * sizeof(float);
        if(wss >= workspace.size())
        {
            workspace.resize(wss);
        }

        // img is this->image. smoothed is an outvar
        // interpolate with det == 1
        // smoothed is an outvar, which is the sampled patch
        // takend from image at the specifeid ellipse
        // FIRST SAMPLE PATCH SHAPE WITHOUT CHANGING SCALES
        Mat smoothed(patchImageSize, patchImageSize, CV_32FC1, (void *)&workspace.front());
        if(!interpolate(img, x, y, a11, a12, a21, a22, smoothed))
        {
            // if interpolate is not touching the image boundary
            // smooth accordingly before sampling to the bigger patch size
            gaussianBlurInplace(smoothed, 1.5f * imageToPatchScale);
            // subsample with corresponding scale
            bool touchesBoundary = interpolate(smoothed,
                                               (float)(patchImageSize >> 1),  // x = half width
                                               (float)(patchImageSize >> 1),  // y = half height
                                               imageToPatchScale, // scale x
                                               0, // no rotation
                                               0, // no shear
                                               imageToPatchScale,  // scale y
                                               this->patch);
            // should have caught a boundary crossing earlier
            assert(!touchesBoundary);
        }
        else
        {
            // interpolate returned true, so we are touching the image boundary
            return true;
        }
    }
    else
    {
        // CASE 2: Smaller patches (in image space)
        // if imageToPatchScale is small (i.e. lot of oversampling), affine normalize without smoothing
        a11 *= imageToPatchScale;
        a12 *= imageToPatchScale;
        a21 *= imageToPatchScale;
        a22 *= imageToPatchScale;
        // ok, do the interpolation
        bool touchesBoundary = interpolate(img, x, y, a11, a12, a21, a22, this->patch);
        assert(!touchesBoundary);
    }
    return false;
}
