/*
 * Copyright (C) 2008-12 Michal Perdoch
 * All rights reserved.
 *
 * This file is part of the HessianAffine detector and is made available under
 * the terms of the BSD license (see the COPYING file).
 *
 */

// Main File. Includes and uses the other files
//

#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/core/core.hpp>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "pyramid.h"
#include "helpers.h"
#include "affine.h"
#include "siftdesc.h"
#include "hesaff.h"

#define make_str(str_name, stream_input) \
            std::string str_name;\
            {std::stringstream tmp_sstm;\
             tmp_sstm << stream_input;\
             str_name = tmp_sstm.str();\
            };

#ifdef MYDEBUG
#undef MYDEBUG
#endif

#ifdef MYDEBUG
#define print(msg) std::cout << msg << std::endl;
#define write(msg) std::cout << msg;
#else
#define print(msg);
#endif

typedef unsigned char uint8;

struct Keypoint
{
   float x, y, s;
   float a11,a12,a21,a22;
   float response;
   int type;
   uint8 desc[128];
};

struct HessianAffineParams
{
   float threshold;
   int   max_iter;
   float desc_factor;
   int   patch_size;
   bool  verbose;
   HessianAffineParams()
      {
         threshold = 16.0f/3.0f;
         max_iter = 16;
         desc_factor = 3.0f*sqrt(3.0f);
         patch_size = 41;
         verbose = false;
      }
};


void rotate_downwards(float &a11, float &a12, float &a21, float &a22)
{
    //same as rectify_up_is_up but doest remove scale
    double a = a11, b = a12, c = a21, d = a22;
    double absdet_ = abs(a * d - b * c);
    double b2a2 = sqrt(b * b + a * a);
    //double sqtdet_ = sqrt(absdet_);
    //-
    a11 = b2a2;
    a12 = 0;
    a21 = (d * b + c * a) / (b2a2);
    a22 = absdet_ / b2a2;
}

void invE_to_invA(cv::Mat& invE, float &a11, float &a12, float &a21, float &a22)
{
    SVD svd_invE(invE, SVD::FULL_UV);
    float *diagE = (float *)svd_invE.w.data;
    diagE[0] = 1.0f / sqrt(diagE[0]);
    diagE[1] = 1.0f / sqrt(diagE[1]);
    // build new invA
    cv::Mat invA_ = svd_invE.u * cv::Mat::diag(svd_invE.w);
    a11 = invA_.at<float>(0,0);
    a12 = invA_.at<float>(0,1);
    a21 = invA_.at<float>(1,0);
    a22 = invA_.at<float>(1,1);
    // Rectify it (maintain scale)
    rotate_downwards(a11, a12, a21, a22);
}


cv::Mat invA_to_invE(float &a11, float &a12, float &a21, float &a22, float& s, float& desc_factor)
{
    float sc = desc_factor * s;
    cv::Mat invA = (cv::Mat_<float>(2,2) << a11, a12, a21, a22);

    //-----------------------
    // Convert invA to invE format
    SVD svd_invA(invA, SVD::FULL_UV);
    float *diagA = (float *)svd_invA.w.data;
    diagA[0] = 1.0f / (diagA[0]*diagA[0]*sc*sc);
    diagA[1] = 1.0f / (diagA[1]*diagA[1]*sc*sc);
    cv::Mat invE = svd_invA.u * cv::Mat::diag(svd_invA.w) * svd_invA.u.t();
    return invE;
}


void fix_A(float &a11, float &a12, float &a21, float &a22, float &s)
{
    // I dont know why we have to convert and convert back
    float desc_factor = 3.0f * sqrt(3.0f); // AffineShape::par.mrSize
    cv::Mat invE = invA_to_invE(a11, a12, a21, a22, s, desc_factor);
    invE_to_invA(invE, a11, a12, a21, a22);
}


struct AffineHessianDetector :
    /*extends*/ public HessianDetector, AffineShape, HessianKeypointCallback, AffineShapeCallback
{
public:
    // Member variables
    const cv::Mat image;
    SIFTDescriptor sift;
    std::vector<Keypoint> keys;

public:
    // Constructor
    AffineHessianDetector(const cv::Mat &image,
                          const PyramidParams &par,
                          const AffineShapeParams &ap,
                          const SIFTDescriptorParams &sp):
        HessianDetector(par), AffineShape(ap), image(image), sift(sp)
    {
        //Inherited from pyramid.h HessianDetector
        this->setHessianKeypointCallback(this);
        // Inherited from affine.h AffineShape
        this->setAffineShapeCallback(this);
    }

    int detect()
    {
        // Reset counters
        this->detectPyramidKeypoints(this->image);
        return this->keys.size();
    }

    void exportArrays(int nKpts, float *kpts, uint8 *desc)
    {
        // Assumes the arrays have been preallocated
        /*  Build output arrays */
        for (size_t fx=0; fx < nKpts; fx++)
        {
            Keypoint &k = keys[fx];
            float x, y, a, b, c, d, s, det;
            float sc = AffineShape::par.mrSize * k.s;
            size_t rowk = fx * 5;
            size_t rowd = fx * 128;
            // given kpts in invA format
            det = k.a11 * k.a22 - k.a12 * k.a21;
            x = k.x;
            y = k.y;
            // Incorporate the scale
            a = sc * k.a11 / (det);
            b = sc * k.a12 / (det);
            c = sc * k.a21 / (det);
            d = sc * k.a22 / (det);

            kpts[rowk + 0] = x;
            kpts[rowk + 1] = y;
            kpts[rowk + 2] = a;
            kpts[rowk + 3] = c;
            kpts[rowk + 4] = d;

            // Assign Descriptor Output
            for (size_t ix = 0; ix < 128; ix++)
            {
                desc[rowd + ix] = uint8(k.desc[ix]);
            }
        }
    }

    void extractDesc(int nKpts, float* kpts, uint8* desc)
        {
        float x, y, ia, ib, ic, id;
        float sc, idet;
        float a11, a12, a21, a22, s;
        for(int fx=0; fx < (nKpts); fx++)
            {
            size_t rowk = fx * 5;
            size_t rowd = fx * 128;
            //Read a keypoint from the file
            x   = kpts[rowk + 0];
            y   = kpts[rowk + 1];
            // We are currently using inv(A) format in HotSpotter
            ia = kpts[rowk + 2];
            ib = 0;
            ic = kpts[rowk + 3];
            id = kpts[rowk + 4];
            idet = abs((ia * id) - (ib * ic));
            // Extract scale.
            sc = sqrt(idet);
            s  = (sc / AffineShape::par.mrSize); // scale
            // Deintegrate scale. Keep invA format
            a11 = ( ia / idet) * sc;
            a12 = 0;  //(ib / idet) / sc;
            a21 = ( ic / idet) * sc;
            a22 = ( id / idet) * sc;
            //rectifyAffineTransformationUpIsUp(a11, a12, a21, a22); //Helper
            if (fx == 0)
                {
                print(x << ", " << y << ", " << a11 << ", " << a12 << ", " << a21  << ", " << a22  << ", " << s)
                }
            // now sample the patch (populates this->patch)
            if (!this->normalizeAffine(this->image, x, y, s, a11, a12, a21, a22)) //affine.cpp
                {
                // compute SIFT descriptor
                this->sift.computeSiftDescriptor(this->patch);
                // Populate output descriptor
                for (int ix=0; ix<128; ix++)
                    {
                    desc[(fx * 128) + ix] = (uint8) this->sift.vec[ix];
                    }
                }
            else
                {
                print("Failure!");
                }
            }
        }

    void write_features(char* img_fpath)
    {
        // Write text keypoints to disk
        char suffix[] = ".hesaff.sift";
        int len = strlen(img_fpath)+strlen(suffix)+1;
        #ifdef WIN32
        char* out_fpath = new char[len];
        #else
        char out_fpath[len];
        #endif
        snprintf(out_fpath, len, "%s%s", img_fpath, suffix); out_fpath[len-1]=0;
        std::ofstream out(out_fpath);
        this->exportKeypoints(out);
        // Clean Up
        #ifdef WIN32
        delete[] out_fpath;
        #endif
    }

    void exportKeypoints(std::ostream &out)
    {
    /*Writes text keypoints in the invE format
     * [iE_a, iE_b]
     * [iE_b, iE_d]
     */
        out << 128 << std::endl;
        int nKpts = keys.size();
        print("Writing " << nKpts << " keypoints");
        out << nKpts << std::endl;
        for (size_t i=0; i<nKpts; i++)
        {
            Keypoint &k = keys[i];
            float sc = AffineShape::par.mrSize * k.s;
            // grab (A) format keypoints (ell->unit)
            cv::Mat A = (cv::Mat_<float>(2,2) << k.a11, k.a12, k.a21, k.a22);
            // Integrate the scale via signular value decomposition
            // Remember
            // A     = U *  S  * V.T
            // invA  = V * 1/S * U.T
            // E     = X *  W  * X.T
            // invE  = Y *  W  * X.T
            // E     = A.T  * A
            // invE  = invA * invA.T
            // X == Y, because E is symmetric
            // W == S^2
            // X == V

            // Decompose A
            SVD svd(A, SVD::FULL_UV);
            float *d = (float *)svd.w.data;
            // Integrate scale into 1/S and take squared inverst to make 1/W
            d[0] = 1.0f / (d[0]*d[0]*sc*sc);
            d[1] = 1.0f / (d[1]*d[1]*sc*sc);
            // Build the matrix invE
            // (I dont understand why U here, but it preserves the rotation I guess)
            // invE = (V * 1/S * U.T) * (U * 1/S * V.T)
            cv::Mat invE = svd.u * cv::Mat::diag(svd.w) * svd.u.t();
            // Write out inv(E)
            out << k.x << " " << k.y << " "
                << invE.at<float>(0,0) << " " << invE.at<float>(0,1) << " " << invE.at<float>(1,1);
            for (size_t i=0; i<128; i++)
                out << " " << int(k.desc[i]);
            out << std::endl;
        }
    }


    void onHessianKeypointDetected(const cv::Mat &blur, float x, float y,
            float s, float pixelDistance,
            int type, float response)
        {
        findAffineShape(blur, x, y, s, pixelDistance, type, response);
        }

    void onAffineShapeFound(const cv::Mat &blur, float x, float y,
            float s, float pixelDistance,
            float a11, float a12,
            float a21, float a22,
            int type, float response,
            int iters)
        {
        // detectPyramidKeypoints -> detectOctaveKeypoints -> localizeKeypoint -> findAffineShape -> onAffineShapeFound
        // convert shape into a up is up frame
        rectifyAffineTransformationUpIsUp(a11, a12, a21, a22); //Helper
        // now sample the patch (populates this->patch)
        if (!normalizeAffine(this->image, x, y, s,
                    a11, a12, a21, a22)) //affine.cpp
            {
            // compute SIFT
            sift.computeSiftDescriptor(this->patch);
            // store the keypoint
            keys.push_back(Keypoint());
            Keypoint &k = keys.back();
            k.x = x; k.y = y; k.s = s;
            k.a11 = a11; k.a12 = a12;
            k.a21 = a21; k.a22 = a22;
            k.response = response;
            k.type = type;
            // store the descriptor
            for (int i=0; i<128; i++)
                {
                k.desc[i] = (uint8) sift.vec[i];
                }
            }
        }


};
// END class AffineHessianDetector

#ifdef __cplusplus
extern "C" {
#endif
typedef void*(*allocer_t)(int, int*);

//http://nbviewer.ipython.org/github/pv/SciPy-CookBook/blob/master/ipython/Ctypes.ipynb

extern HESAFF_EXPORT int detect(AffineHessianDetector* detector)
{
    print("detector->detect");
    int nKpts = detector->detect();
    print("nKpts = " << nKpts);
    return nKpts;
}

extern HESAFF_EXPORT AffineHessianDetector* new_hesaff(char* img_fpath)
{
    print("making detector for " << img_fpath);
    print("make hesaff. img_fpath = " << img_fpath);
    // Read in image and convert to uint8
    cv::Mat tmp = cv::imread(img_fpath);
    cv::Mat image(tmp.rows, tmp.cols, CV_32FC1, Scalar(0));
    float *imgout = image.ptr<float>(0);
    uint8 *imgin  = tmp.ptr<uint8>(0);
    for (size_t i=tmp.rows*tmp.cols; i > 0; i--)
    {
        *imgout = (float(imgin[0]) + imgin[1] + imgin[2])/3.0f;
        imgout++;
        imgin+=3;
    }
    // Define params
    HessianAffineParams par;
    SIFTDescriptorParams siftParams;
    PyramidParams pyrParams;
    AffineShapeParams affShapeParams;
    // copy params
    pyrParams.threshold          = par.threshold;
    affShapeParams.maxIterations = par.max_iter;
    affShapeParams.patchSize     = par.patch_size;
    affShapeParams.mrSize        = par.desc_factor;
    siftParams.patchSize         = par.patch_size;
    // Execute detection
    int nKpts;
    AffineHessianDetector* detector = new AffineHessianDetector(image, pyrParams, affShapeParams, siftParams);
    return detector;
}

extern HESAFF_EXPORT void extractDesc(AffineHessianDetector* detector, int nKpts, float* kpts, uint8* desc)
{
    print("detector->extractDesc");
    detector->extractDesc(nKpts, kpts, desc);
    print("nKpts = " << nKpts);
}

extern HESAFF_EXPORT void exportArrays(AffineHessianDetector* detector, int nKpts, float *kpts, uint8 *desc)
{
    print("detector->exportArrays(" << nKpts << ")");
    print("detector->exportArrays kpts[0]" << kpts[0] << ")");
    print("detector->exportArrays desc[0]" << desc[0] << ")");
    detector->exportArrays(nKpts, kpts, desc);
    print("detector->exportArrays kpts[0]" << kpts[0] << ")");
    print("detector->exportArrays desc[0]" << desc[0] << ")");
    print("FINISHED detector->exportArrays");
}

extern HESAFF_EXPORT void writeFeatures(AffineHessianDetector* detector, char* img_fpath)
{
    print("detector->write_features");
    detector->write_features(img_fpath);
}

#ifdef __cplusplus
}
#endif

int main(int argc, char **argv)
{
    if (argc>1)
        {
        char* img_fpath = argv[1];
        int nKpts;
        AffineHessianDetector* detector = new_hesaff(img_fpath);
        nKpts = detect(detector);
        writeFeatures(detector, img_fpath);
        }
   else
       {
       printf("\nUsage: ell_desc image_name.png kpts_file.txt\nDescribes elliptical keypoints (with gravity vector) given in kpts_file.txt using a SIFT descriptor.\n\n");
       }
}
