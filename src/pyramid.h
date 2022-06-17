/*
 * Copyright (C) 2008-12 Michal Perdoch
 * All rights reserved.
 *
 * This file is part of the HessianAffine detector and is made available under
 * the terms of the BSD license (see the COPYING file).
 *
 */

#ifndef __PYRAMID_H__
#define __PYRAMID_H__

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;

struct PyramidParams
{
    // number of scale per octave
    int  numberOfScales;
    // amount of smoothing applied to the initial level of first octave
    float initialSigma;
    // noise dependent threshold on the response (sensitivity)
    float threshold;
    // ratio of the eigenvalues
    float edgeEigenValueRatio;
    // number of pixels ignored at the border of image
    int  border;
    int  maxPyramidLevels;

    // DENSE DETECTION PARAMS
    bool use_dense;
    int dense_stride;

    PyramidParams()
    {
        numberOfScales = 3;
        initialSigma = 1.6f;
        threshold = 16.0f/3.0f; //0.04f * 256 / 3;
        edgeEigenValueRatio = 10.0f;
        border = 5;
        maxPyramidLevels = -1;

        use_dense = false;
        dense_stride = 32;
    }
};

struct HessianKeypointCallback
{
    virtual void onHessianKeypointDetected(
            const cv::Mat &blur,
            float x, float y,
            float s,
            float pixelDistance,
            int type, float response) = 0;
};

struct HessianDetector
{
    enum {
        HESSIAN_DARK   = 0,
        HESSIAN_BRIGHT = 1,
        HESSIAN_SADDLE = 2,
    };
public:
    HessianKeypointCallback *hessianKeypointCallback;
    PyramidParams par;
    HessianDetector(const PyramidParams &par) :
        edgeScoreThreshold((par.edgeEigenValueRatio + 1.0f)*(par.edgeEigenValueRatio + 1.0f)/par.edgeEigenValueRatio),
        // thresholds are squared, response of det H is proportional to square of derivatives!
        finalThreshold(par.threshold * par.threshold),
        positiveThreshold(0.8f * finalThreshold),
        negativeThreshold(-positiveThreshold)
    {
        this->par = par;
        hessianKeypointCallback = 0;
    }
    void setHessianKeypointCallback(HessianKeypointCallback *callback)
    {
        hessianKeypointCallback = callback;
    }
    void detectPyramidKeypoints(const cv::Mat &image);

protected:
    void detectOctaveHessianKeypoints(const cv::Mat &firstLevel, float pixelDistance, cv::Mat &nextOctaveFirstLevel);
    void detectOctaveDenseKeypoints(const cv::Mat &firstLevel, float pixelDistance, cv::Mat &nextOctaveFirstLevel);
    void localizeKeypoint(int r, int c, float curScale, float pixelDistance);
    void findLevelKeypoints(float curScale, float pixelDistance);
    void findDenseLevelKeypoints(float curScale, float pixelDistance); // hacked in
    cv::Mat hessianResponse(const cv::Mat &inputImage, float norm);

private:
    // some constants derived from parameters
    const float edgeScoreThreshold;
    const float finalThreshold;
    const float positiveThreshold;
    const float negativeThreshold;

    // temporary arrays used by protected functions
    cv::Mat octaveMap;
    cv::Mat prevBlur, blur;
    cv::Mat low, cur, high;
};

#endif // __PYRAMID_H__
