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

#ifdef WIN32
#define M_PI 3.14159
#endif

// The SIFT descriptor is subject to US Patent 6,711,293

void SIFTDescriptor::precomputeBinsAndWeights()
{
   int halfSize = this->par.patchSize>>1;
   float step = float(this->par.spatialBins+1)/(2*halfSize);
   
   // allocate maps at the same location
   this->precomp_bins.resize(2*this->par.patchSize);
   this->precomp_weights.resize(2*this->par.patchSize);
   this->bin1 = this->bin0 = &this->precomp_bins.front(); this->bin1 += this->par.patchSize;
   this->w1   =   this->w0 = &this->precomp_weights.front(); this->w1 += this->par.patchSize;

   // maps every pixel in the patch 0..patch_size-1 to appropriate spatial bin and weight
   for (int i = 0; i < this->par.patchSize; i++)
   {
      float x = step*i;      // x goes from <-1 ... spatial_bins> + 1
      int  xi = (int)(x);
      // bin indices
      this->bin0[i] = xi-1; // get real xi
      this->bin1[i] = xi;
      // weights
      this->w1[i]   = x - xi;
      this->w0[i]   = 1.0f - this->w1[i];
      // truncate weights and bins in case they reach outside of valid range
      if (this->bin0[i] <          0) { this->bin0[i] = 0;           this->w0[i] = 0; }
      if (this->bin0[i] >= this->par.spatialBins) { this->bin0[i] = this->par.spatialBins-1; this->w0[i] = 0; }
      if (this->bin1[i] <          0) { this->bin1[i] = 0;           this->w1[i] = 0; }
      if (this->bin1[i] >= this->par.spatialBins) { this->bin1[i] = this->par.spatialBins-1; this->w1[i] = 0; }
      // adjust for orientation bin skip
      this->bin0[i] *= this->par.orientationBins;
      this->bin1[i] *= this->par.orientationBins;
   }
}

void SIFTDescriptor::samplePatch()
{  
   for (int r = 0; r < this->par.patchSize; ++r)
   {
      const int br0 = this->par.spatialBins * this->bin0[r]; const float wr0 = this->w0[r];
      const int br1 = this->par.spatialBins * this->bin1[r]; const float wr1 = this->w1[r];
      for (int c = 0; c < this->par.patchSize; ++c)
      {
         float val = this->mask.at<float>(r,c) * this->grad.at<float>(r,c);
         
         const int bc0 = this->bin0[c]; const float wc0 = this->w0[c]*val;
         const int bc1 = this->bin1[c]; const float wc1 = this->w1[c]*val;
         
         // ori from atan2 is in range <-pi,pi> so add 2*pi to be surely above zero         
         const float o = float(this->par.orientationBins)*(this->ori.at<float>(r,c) + 2*M_PI)/(2*M_PI);
         
         int   bo0 = (int)o;
         const float wo1 =  o - bo0; 
         bo0 %= this->par.orientationBins;
         
         int   bo1 = (bo0+1) % this->par.orientationBins; 
         const float wo0 = 1.0f - wo1;
         
         // add to corresponding 8 vec...
         val = wr0*wc0; if (val>0) { this->vec[br0+bc0+bo0] += val * wo0; this->vec[br0+bc0+bo1] += val * wo1; }
         val = wr0*wc1; if (val>0) { this->vec[br0+bc1+bo0] += val * wo0; this->vec[br0+bc1+bo1] += val * wo1; }
         val = wr1*wc0; if (val>0) { this->vec[br1+bc0+bo0] += val * wo0; this->vec[br1+bc0+bo1] += val * wo1; }
         val = wr1*wc1; if (val>0) { this->vec[br1+bc1+bo0] += val * wo0; this->vec[br1+bc1+bo1] += val * wo1; }
      }
   }
}

float SIFTDescriptor::normalize()
{
   float vectlen = 0.0f;
   for (size_t i = 0; i < this->vec.size(); i++) 
   {
      const float val = this->vec[i];
      vectlen += val * val;
   }
   vectlen = sqrt(vectlen);
   
   const float fac = float(1.0f / vectlen);
   for (size_t i = 0; i < this->vec.size(); i++) this->vec[i] *= fac;
   return vectlen;
}

void SIFTDescriptor::sample()
{
   for (size_t i = 0; i < this->vec.size(); i++) this->vec[i]=0;
   // accumulate histograms
   this->samplePatch(); normalize();
   // check if there are some values above threshold
   bool changed = false; 
   for (size_t i = 0; i < this->vec.size(); i++) if (this->vec[i] > this->par.maxBinValue) { this->vec[i] = this->par.maxBinValue; changed = true; }
   if (changed) normalize();

   for (size_t i = 0; i < this->vec.size(); i++) 
   {
      int b  = min((int)(512.0f * this->vec[i]), 255);
      this->vec[i] = float(b);
   }
}

void SIFTDescriptor::computeSiftDescriptor(Mat &patch)
{
   const int width = patch.cols;
   const int height = patch.rows;
   // photometrically normalize with weights as in SIFT gradient magnitude falloff
   float mean, var;
   photometricallyNormalize(patch, this->mask, mean, var);
   // prepare gradients
   for (int r = 0; r < height; ++r)
      for (int c = 0; c < width; ++c) 
      {
         float xgrad, ygrad; 
         if (c == 0) xgrad = patch.at<float>(r,c+1) - patch.at<float>(r,c); else 
            if (c == width-1) xgrad = patch.at<float>(r,c) - patch.at<float>(r,c-1); else 
               xgrad = patch.at<float>(r,c+1) - patch.at<float>(r,c-1);
         
         if (r == 0) ygrad = patch.at<float>(r+1,c) - patch.at<float>(r,c); else 
            if (r == height-1) ygrad = patch.at<float>(r,c) - patch.at<float>(r-1,c); else
               ygrad = patch.at<float>(r+1,c) - patch.at<float>(r-1,c);
         
         this->grad.at<float>(r,c) = ::sqrt(xgrad * xgrad + ygrad * ygrad);
         this->ori.at<float>(r,c) = atan2(ygrad, xgrad);
      }
   // compute SIFT vector
   this->sample();
}
