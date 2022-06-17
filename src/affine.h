/*
 * Copyright (C) 2008-12 Michal Perdoch
 * All rights reserved.
 *
 * This file is part of the HessianAffine detector and is made available under
 * the terms of the BSD license (see the COPYING file).
 *
 */

#ifndef __AFFINE_H__
#define __AFFINE_H__

#include <vector>
#include <opencv2/opencv.hpp>
#include "helpers.h"


//----------------------
// Params Struct
//----------------------
struct AffineShapeParams
{
    int maxIterations; // number of affine shape interations
    float convergenceThreshold; // max deviation from isotropic shape to converge
    int smmWindowSize;   // width and height of the SMM mask
    int patchSize;       // width and height of the patch
    float initialSigma;  // amount of smoothing applied to the initial level of first octave
    float mrSize;        // size of the measurement region (as multiple of the feature scale)

    AffineShapeParams()
    {
        maxIterations = 16;
        initialSigma = 1.6f;
        convergenceThreshold = 0.05f;
        patchSize = 41;
        smmWindowSize = 19;
        mrSize = 3.0f*sqrt(3.0f);
    }
};


//----------------------
// Callback Struct
//----------------------
struct AffineShapeCallback
{
    virtual void onAffineShapeFound(
        const cv::Mat &blur,     // corresponding scale level
        float x, float y,     // subpixel, image coordinates
        float s,              // scale
        float pixelDistance,  // distance between pixels in provided blured image
        float a11, float a12, // affine shape matrix
        float a21, float a22,
        int type, float response, int iters) = 0;
};

//----------------------
// Computing Struct
//----------------------
struct AffineShape
{
public:
    //---------------------
    // Standard functions
    //---------------------
    AffineShape(const AffineShapeParams &par) :
        patch(par.patchSize, par.patchSize, CV_32FC1),
        mask(par.smmWindowSize, par.smmWindowSize, CV_32FC1),
        img(par.smmWindowSize, par.smmWindowSize, CV_32FC1),
        fx(par.smmWindowSize, par.smmWindowSize, CV_32FC1),
        fy(par.smmWindowSize, par.smmWindowSize, CV_32FC1)
    {
        this->par = par;
        computeGaussMask(mask);  // Defined in helpers.cpp
        affineShapeCallback = 0;
        fx = cv::Scalar(0);
        fy = cv::Scalar(0);
    }

    ~AffineShape() {}

    void setAffineShapeCallback(AffineShapeCallback *callback)
    {
        affineShapeCallback = callback;
    }

    //---------------------
    // Work functions
    //---------------------

    // computes affine shape
    bool findAffineShape(const cv::Mat &blur, float x, float y, float s,
                         float pixelDistance, int type, float response);

    // fills patch with affine normalized neighbourhood around point in the img, enlarged mrSize times
    bool normalizeAffine(const cv::Mat &img, float x, float y, float s,
                         float a11, float a12, float a21, float a22, float ori);

    // Checks if we could fill affine neighborhood
    bool normalizeAffineCheckBorders(const cv::Mat &img, float x, float y, float s,
                                     float a11, float a12, float a21, float
                                     a22, float ori);

public:
    cv::Mat patch;  // member var to store a computed patch in

protected:
    AffineShapeParams par;

private:
    AffineShapeCallback *affineShapeCallback;
    std::vector<unsigned char> workspace;
    cv::Mat mask, img, fx, fy;
};

#endif // __AFFINE_H__
