/*
 * Copyright (C) 2008-12 Michal Perdoch
 * All rights reserved.
 *
 * This file is part of the HessianAffine detector and is made available under
 * the terms of the BSD license (see the COPYING file).
 *
 */

#include <vector>
#include "siftdesc.h"

using namespace std;
using namespace cv;

#ifndef M_PI
#define M_PI 3.14159f
#endif

//#define DEBUG_SIFT 1
#define DEBUG_SIFT 0

#if DEBUG_SIFT
    #define printDBG_SIFT(msg) std::cerr << "[sift.c] " << msg << std::endl;
#else
    #define printDBG_SIFT(msg);
#endif

// The SIFT descriptor is subject to US Patent 6,711,293

SIFTDescriptor::SIFTDescriptor(const SIFTDescriptorParams &par) :
    mask(par.patchSize, par.patchSize, CV_32FC1),
    grad(par.patchSize, par.patchSize, CV_32FC1),
    ori(par.patchSize, par.patchSize, CV_32FC1)
{
    this->par = par;
    this->vec.resize(par.spatialBins * par.spatialBins * par.orientationBins);
    computeCircularGaussMask(this->mask);  // defined in helpers.cpp
    this->precomputeBinsAndWeights();
}


void SIFTDescriptor::initialize()
{
    for(size_t i = 0; i < this->vec.size(); i++)
    {
        this->vec[i] = 0;
    }
}


void SIFTDescriptor::precomputeBinsAndWeights()
{
    int halfSize = this->par.patchSize >> 1;
    float step = float(this->par.spatialBins + 1) / (2 * halfSize);

    // allocate maps at the same location
    this->precomp_bins.resize(2 * this->par.patchSize);
    this->precomp_weights.resize(2 * this->par.patchSize);
    this->bin1 = this->bin0 = &this->precomp_bins.front();
    this->bin1 += this->par.patchSize;
    this->w1 = this->w0 = &this->precomp_weights.front();
    this->w1 += this->par.patchSize;

    // maps every pixel in the patch 0..patch_size-1 to appropriate spatial bin and weight
    for(int i = 0; i < this->par.patchSize; i++)
    {
        float x = step * i;    // x goes from <-1 ... spatial_bins> + 1
        int  xi = (int)(x);
        // bin indices
        this->bin0[i] = xi - 1; // get real xi
        this->bin1[i] = xi;
        // weights
        this->w1[i]   = x - xi;
        this->w0[i]   = 1.0f - this->w1[i];
        // truncate weights and bins in case they reach outside of valid range
        if(this->bin0[i] < 0)
        {
            this->bin0[i] = 0;
            this->w0[i] = 0;
        }
        if(this->bin0[i] >= this->par.spatialBins)
        {
            this->bin0[i] = this->par.spatialBins - 1;
            this->w0[i] = 0;
        }
        if(this->bin1[i] < 0)
        {
            this->bin1[i] = 0;
            this->w1[i] = 0;
        }
        if(this->bin1[i] >= this->par.spatialBins)
        {
            this->bin1[i] = this->par.spatialBins - 1;
            this->w1[i] = 0;
        }
        // adjust for orientation bin skip
        this->bin0[i] *= this->par.orientationBins;
        this->bin1[i] *= this->par.orientationBins;
    }
}

void SIFTDescriptor::samplePatch()
{
    // Measure gradient information and record in
    // the weighted oriented histogram
    for(int r = 0; r < this->par.patchSize; ++r)
    {
        const int br0 = this->par.spatialBins * this->bin0[r];
        const float wr0 = this->w0[r];
        const int br1 = this->par.spatialBins * this->bin1[r];
        const float wr1 = this->w1[r];
        for(int c = 0; c < this->par.patchSize; ++c)
        {
            float val = this->mask.at<float>(r, c) * this->grad.at<float>(r, c);

            const int bc0 = this->bin0[c];
            const float wc0 = this->w0[c] * val;
            const int bc1 = this->bin1[c];
            const float wc1 = this->w1[c] * val;

            // ori from atan2 is in range <-pi,pi> so add 2*pi to be surely above zero
            const float o = float(this->par.orientationBins) * (this->ori.at<float>(r, c) + 2.0f * M_PI) / (2.0f * M_PI);

            int   bo0 = (int)o;
            const float wo1 =  o - bo0;
            bo0 %= this->par.orientationBins;

            int   bo1 = (bo0 + 1) % this->par.orientationBins;
            const float wo0 = 1.0f - wo1;

            // add to corresponding 8 vec...
            val = wr0 * wc0;
            if(val > 0)
            {
                this->vec[br0 + bc0 + bo0] += val * wo0;
                this->vec[br0 + bc0 + bo1] += val * wo1;
            }
            val = wr0 * wc1;
            if(val > 0)
            {
                this->vec[br0 + bc1 + bo0] += val * wo0;
                this->vec[br0 + bc1 + bo1] += val * wo1;
            }
            val = wr1 * wc0;
            if(val > 0)
            {
                this->vec[br1 + bc0 + bo0] += val * wo0;
                this->vec[br1 + bc0 + bo1] += val * wo1;
            }
            val = wr1 * wc1;
            if(val > 0)
            {
                this->vec[br1 + bc1 + bo0] += val * wo0;
                this->vec[br1 + bc1 + bo1] += val * wo1;
            }
        }
    }
}


//float SIFTDescriptor::normp(const float& p)
//{
//    // Compute the LP norm
//    float ellp_norm = 0.0f;
//    for(size_t i = 0; i < this->vec.size(); i++)
//    {
//        const float val = this->vec[i];
//        ellp_norm += ::pow(::abs(val), p)
//    }
//    ellp_norm = ::pow(ellp_norm, 1.0 / p);
//    return ellp_norm;
//}

//float SIFTDescriptor::norm_inf()
//{
//    // Compute the Linfinity norm
//    float ellinf_norm = 0.0f;
//    for(size_t i = 0; i < this->vec.size(); i++)
//    {
//        const float val = ::abs(this->vec[i]);
//        if (val > ellinf_norm){
//            ellinf_norm = val
//        }
//    }
//    return ellinf_norm;
//}


float SIFTDescriptor::norm2()
{
    // Compute the L2 norm
    float ell2_norm = 0.0f;
    // Compute norm_ = sqrt((vec ** 2).sum())
    for(size_t i = 0; i < this->vec.size(); i++)
    {
        const float val = this->vec[i];
        ell2_norm += val * val;
    }
    ell2_norm = sqrt(ell2_norm);
    return ell2_norm;
}


float SIFTDescriptor::norm1()
{
    // Compute the L1 norm
    float ell1_norm = 0.0f;
    // Compute norm_ = abs(vec).sum()
    for(size_t i = 0; i < this->vec.size(); i++)
    {
        ell1_norm += ::fabs(this->vec[i]);
    }
    return ell1_norm;
}


float SIFTDescriptor::normalize2()
{
    // Do L2 normalization
    float ell2_norm = this->norm2();

    // Compute vec /= norm_
    const float fac = float(1.0f / ell2_norm);
    for(size_t i = 0; i < this->vec.size(); i++)
    {
        this->vec[i] *= fac;
    }
    return ell2_norm;
}


float SIFTDescriptor::normalize1()
{
    // Do L1 normalization
    float ell1_norm = this->norm1();

    // Compute vec /= norm_
    const float fac = float(1.0f / ell1_norm);
    for(size_t i = 0; i < this->vec.size(); i++)
    {
        this->vec[i] *= fac;
    }
    return ell1_norm;
}


void SIFTDescriptor::powerLaw()
{
    // Apply any power law transformation
    // Handle special case (RootSIFT) of power=.5; sqrt is faster than pow
    if (this->par.siftPower == 0.5)
    {
        for(size_t i = 0; i < this->vec.size(); i++)
        {
            this->vec[i] = ::sqrt(this->vec[i]);
        }
    }
    // apply more general power law
    else
    {
        for(size_t i = 0; i < this->vec.size(); i++)
        {
            this->vec[i] = ::pow(this->vec[i], this->par.siftPower);
        }
    }
}


bool SIFTDescriptor::clipBins()
{
    bool changed = false;
    // check if there are some descriptor values above threshold
    for(size_t i = 0; i < this->vec.size(); i++)
    {
        if(this->vec[i] > this->par.maxBinValue)
        {
            this->vec[i] = this->par.maxBinValue;
            changed = true;
        }
    }
    return changed;
}


void SIFTDescriptor::quantize()
{
    // Quantize into range 0-255 but use a hack
    for(size_t i = 0; i < this->vec.size(); i++)
    {
        // Tricky: Components are likely to be less than .5 due to L2.
        // Therefore it is usually safe to multiply by 512.0, which also gives
        // the uint8s more fidelity and reduces quantization error.
        int b = min((int)(512.0f * this->vec[i]), 255);
        this->vec[i] = float(b);
    }
}



void SIFTDescriptor::sample()
{
    /*
     * Computes this->vec (The 128D SIFT descriptor) of an image patch
     */

    // TODO: return original vector length
    // then use this to filter out homogenous keypoints as in
    // A comparison of dense region detectors for image search and fine-grained
    // classification (2015)

    // Initialize descriptor vector to zero
    this->initialize();

    // accumulate histograms
    this->samplePatch();

    if (this->par.siftPower != 1.0)
    {
        // Apply power law / (root SIFT)
        // NOTE: We do not need to apply L1 normalization before the power law
        // if we are applying L2 normalization after it.
        this->powerLaw();
    }

    // Do L2 normalization after any power law
    float orig_mag = this->normalize2();

    if (this->par.maxBinValue >= 0.0)
    {
        // Apply clipping procedure (normal SIFT)
        bool changed = this->clipBins();
        // L2 normalize descriptor vector again if it was clipped
        if(changed)
        {
            this->normalize2();
        }
    }

    this->quantize();
}

    //#if DEBUG_SIFT
    //    //printDBG_SIFT("changed " << changed);
    //    printDBG_SIFT("this->par.maxBinValue " << this->par.maxBinValue);
    //    float maxval_postclip = *max_element(this->vec.begin(), this->vec.end());
    //    printDBG_SIFT("maxval_postclip " << maxval_postclip);
    //    printDBG_SIFT("])");
    //#endif

void SIFTDescriptor::computeSiftDescriptor(Mat &patch)
{
    const int width = patch.cols;
    const int height = patch.rows;
    // photometrically normalize with weights as in SIFT gradient magnitude falloff
    float mean, var;
    // this->mask is computed in the constructor
    photometricallyNormalize(patch, this->mask, mean, var);
    // prepare gradients
    for(int r = 0; r < height; ++r)
        for(int c = 0; c < width; ++c)
        {
            float xgrad, ygrad;
            if(c == 0)
            {
                xgrad = patch.at<float>(r, c + 1) - patch.at<float>(r, c);
            }
            else if(c == width - 1)
            {
                xgrad = patch.at<float>(r, c) - patch.at<float>(r, c - 1);
            }
            else
            {
                xgrad = patch.at<float>(r, c + 1) - patch.at<float>(r, c - 1);
            }

            if(r == 0)
            {
                ygrad = patch.at<float>(r + 1, c) - patch.at<float>(r, c);
            }
            else if(r == height - 1)
            {
                ygrad = patch.at<float>(r, c) - patch.at<float>(r - 1, c);
            }
            else
            {
                ygrad = patch.at<float>(r + 1, c) - patch.at<float>(r - 1, c);
            }

            this->grad.at<float>(r, c) = ::sqrt(xgrad * xgrad + ygrad * ygrad);
            this->ori.at<float>(r, c) = atan2(ygrad, xgrad);
        }
    // compute SIFT vector
    this->sample();
}
