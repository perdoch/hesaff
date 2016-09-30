/*
 * Copyright (C) 2008-12 Michal Perdoch
 * All rights reserved.
 *
 * This file is part of the HessianAffine detector and is made available under
 * the terms of the BSD license (see the COPYING file).
 *
 */

// The SIFT descriptor is subject to US Patent 6,711,293

#ifndef __SIFTDESC_H__
#define __SIFTDESC_H__

#include <vector>

#include <opencv2/opencv.hpp>

#include "helpers.h"

struct SIFTDescriptorParams
{
    int spatialBins;
    int orientationBins;
    float maxBinValue;
    int patchSize;
    float siftPower;
    SIFTDescriptorParams()
    {
        spatialBins = 4;
        orientationBins = 8;
        maxBinValue = 0.2f;  // clipping
        patchSize = 41;
        siftPower = 1.0;  
        // TODO: mean vector?
        //
        // Common Variants:
        // Lowe's Original SIFT: (defaults)
        // ROOT SIFT: siftPower=.5,maxBinValue=-1
    }
};


struct SIFTDescriptor
{

public:
    
    // top level interface
    SIFTDescriptor(const SIFTDescriptorParams &par); 

    void computeSiftDescriptor(cv::Mat &patch);

public:
    std::vector<float> vec;

private:
    // helper functions

    void sample();
    void samplePatch();
    void precomputeBinsAndWeights();

    float norm1();
    float norm2();
    float normalize1();
    float normalize2();

    void powerLaw();
    bool clipBins();
    void initialize();
    void quantize();

private:
    SIFTDescriptorParams par;
    cv::Mat mask, grad, ori;
    std::vector<int> precomp_bins;
    std::vector<float> precomp_weights;
    int *bin0, *bin1;
    float *w0, *w1;
};

#endif //__SIFTDESC_H__
