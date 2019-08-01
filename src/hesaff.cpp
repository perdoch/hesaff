int global_nkpts = 0;
int global_nmulti_ori = 0;
int global_c1 = 0;
int global_c2 = 0;
/*
 * Copyright (C) 2008-12 Michal Perdoch
 * All rights reserved.
 *
 * This file is part of the HessianAffine detector and is made available under
 * the terms of the BSD license (see the COPYING file).
 *
 */
// Modifications by Jon Crall
/*
 *
CommandLine:
    mingw_build.bat && python -c "import utool as ut; ut.cmd('build/hesaffexe.exe ' + ut.grab_test_imgpath('star.png'))"
    ./unix_build.sh && python -c "import utool as ut; ut.cmd('build/hesaffexe ' + ut.grab_test_imgpath('star.png'))"

    python -m pyhesaff detect_feats --fname lena.png --verbose  --show  --rebuild-hesaff --no-rmbuild

    python -m pyhesaff._pyhesaff --test-test_rot_invar --show --rebuild-hesaff --no-rmbuild
    python -m pyhesaff._pyhesaff --test-test_rot_invar --show

    astyle --style=ansi --indent=spaces  --indent-classes  --indent-switches \
        --indent-col1-comments --pad-oper --unpad-paren --delete-empty-lines \
        --add-brackets *.cpp *.h
 */

// Main File. Includes and uses the other files
//

#define DEBUG_HESAFF 0

#include <iostream>
#include <fstream>
#include <string>
#if DEBUG_HESAFF
#include <assert.h>
#endif
#include <opencv2/core.hpp>

#define USE_FREAK 0

#ifdef USE_FREAK
//#include <opencv2/nonfree.hpp>
#endif

//#include <opencv2/core/utility.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <stdlib.h> // malloc
#include <string.h> // strcpy

#include "pyramid.h"
#include "helpers.h"
#include "orientation.h"
#include "affine.h"
#include "siftdesc.h"
#include "hesaff.h"

#define make_str(str_name, stream_input) \
    std::string str_name;\
{std::stringstream tmp_sstm;\
    tmp_sstm << stream_input;\
    str_name = tmp_sstm.str();\
};

#ifndef M_PI
#define M_PI 3.14159f
#endif

#ifndef M_TAU
#define M_TAU 6.28318f
#endif

// Gravity points downward = tau / 4 = pi / 2
#ifndef M_GRAVITY_THETA
#define M_GRAVITY_THETA 1.570795f
// relative to gravity
#define R_GRAVITY_THETA 0.0f
#endif

#if DEBUG_HESAFF
    #define printDBG(msg) std::cout << "[hesaff.c] " << msg << std::endl;
    #define write(msg) std::cout << msg;
#else
    #define printDBG(msg);
#endif


#define USE_ORI 1  // developing rotational invariance


#if USE_ORI
static const int KPTS_DIM = 6;
#else
static const int KPTS_DIM = 5;
#endif
static const int DESC_DIM = 128;

typedef unsigned char uint8;

struct Keypoint
{
    float x, y, s;
    float a11, a12, a21, a22;
    #if USE_ORI
    float ori;
    #endif
    float response;
    int type;
    uint8 desc[DESC_DIM];
};

extern void computeGradient(const cv::Mat &img, cv::Mat &gradx, cv::Mat &grady); //from affine.cpp

struct AffineHessianDetector : public HessianDetector, AffineShape, HessianKeypointCallback, AffineShapeCallback
{
public:
    // Member variables
    const cv::Mat image;
    SIFTDescriptor sift;
    std::vector<Keypoint> keys;
    int num_kpts;
    const HesaffParams hesPar;
public:
    // Constructor
    AffineHessianDetector(const cv::Mat &image,
                          const PyramidParams &par,
                          const AffineShapeParams &ap,
                          const SIFTDescriptorParams &sp,
                          const HesaffParams& hesParams):
        HessianDetector(par), AffineShape(ap), image(image), sift(sp), hesPar(hesParams)
    {
        this->num_kpts = 0;
        this->setHessianKeypointCallback(this); //Inherits from pyramid.h HessianDetector
        this->setAffineShapeCallback(this); // Inherits from affine.h AffineShape
    }

    int detect()
    {
        // Reset counters
        this->detectPyramidKeypoints(this->image);
        #if DEBUG_HESAFF
        if (!hesPar.only_count)
        {
            assert(this->num_kpts == this->keys.size());
        }
        #endif
        return this->num_kpts;
        //return this->keys.size();
    }

    void exportArrays(int nKpts, float *kpts, uint8 *desc)
    {
        // Exports keypoints and descriptors into preallocated numpy arrays
        for(size_t fx = 0; fx < nKpts; fx++)
        {
            Keypoint &k = this->keys[fx];
            float x, y, iv11, iv12, iv21, iv22, det;
            const float sc = AffineShape::par.mrSize * k.s;
            const size_t rowk = fx * KPTS_DIM;
            const size_t rowd = fx * DESC_DIM;
            // given kpts in invV format
            det = (k.a11 * k.a22) - (k.a12 * k.a21);
            x = k.x;
            y = k.y;
            // Incorporate the scale
            iv11 = sc * k.a11 / (det);
            iv12 = sc * k.a12 / (det);
            iv21 = sc * k.a21 / (det);
            iv22 = sc * k.a22 / (det);

            kpts[rowk + 0] = x;
            kpts[rowk + 1] = y;
            kpts[rowk + 2] = iv11;
            kpts[rowk + 3] = iv21;
            kpts[rowk + 4] = iv22;
            #if USE_ORI
            kpts[rowk + 5] = k.ori;
            #endif

            // Assign Descriptor Output
            for(size_t ix = 0; ix < DESC_DIM; ix++)
            {
                desc[rowd + ix] = uint8(k.desc[ix]);
            }
        }
    }

    void exportKeypoints(std::ostream &out)
    {
        /* Writes text keypoints in the invE format to a stdout stream
         *  [iE_a, iE_b]
         *  [iE_b, iE_d]
         */
        out << DESC_DIM << std::endl;
        int nKpts = static_cast<int>(this->keys.size());
        printDBG("[export] Writing " << nKpts << " keypoints");
        out << nKpts << std::endl;
        for(size_t i = 0; i < nKpts; i++)
        {
            Keypoint &k = this->keys[i];
            const float sc = AffineShape::par.mrSize * k.s;
            // Grav invA keypoints
            cv::Mat invA = (cv::Mat_<float>(2, 2) << k.a11, k.a12, k.a21, k.a22);
            // Integrate the scale via signular value decomposition
            // Remember
            // A     = U *  S  * V.T   // SVD Step
            // invA  = V * 1/S * U.T   // Linear Algebra
            // E     = X *  W  * X.T
            // invE  = Y *  W  * X.T
            // E     = A.T  * A
            // invE  = invA * invA.T
            // X == Y, because E is symmetric
            // W == S^2
            // X == V
            // Decompose invA
            SVD svd_invA(invA, SVD::FULL_UV);
            float * const diag_invA = (float *)svd_invA.w.data;
            // Integrate scale into 1/S and take squared inverst to make 1/W
            diag_invA[0] = 1.0f / (diag_invA[0] * diag_invA[0] * sc * sc);
            diag_invA[1] = 1.0f / (diag_invA[1] * diag_invA[1] * sc * sc);
            // Build the matrix invE
            // (I dont understand why U is here, but it preserves the rotation I guess)
            // invE = (V * 1/S * U.T) * (U * 1/S * V.T)
            cv::Mat invE = svd_invA.u * cv::Mat::diag(svd_invA.w) * svd_invA.u.t();
            // Write inv(E) to out stream
            const float e11 = invE.at<float>(0, 0);
            const float e12 = invE.at<float>(0, 1); // also e12 because of E symetry
            const float e22 = invE.at<float>(1, 1);
            #if USE_ORI
            float ori = k.ori;
            out << k.x << " " << k.y << " "
                << e11 << " " << e12 << " "
                << e22 << " " << ori;
            #else
            out << k.x << " " << k.y << " "
                << e11 << " " << e12 << " " << e22 ;
            #endif
            for(size_t i = 0; i < DESC_DIM; i++)
            {
                out << " " << int(k.desc[i]);
            }
            out << std::endl;
        }
    }


    void onHessianKeypointDetected(const cv::Mat &blur, float x, float y,
                                   float s, float pixelDistance,
                                   int type, float response)
    {
        if (hesPar.affine_invariance)
        {
            // A circular keypoint is detected. Adpat its shape to an ellipse
            findAffineShape(blur, x, y, s, pixelDistance, type, response);
        }
        else
        {
            // Otherwise just represent the circle as an ellipse
            //float eigval1 = 1.0f, eigval2 = 1.0f;
            //float lx = x / pixelDistance, ly = y / pixelDistance;
            //float ratio = s / (par.initialSigma * pixelDistance);
            float u11 = 1.0f, u12 = 0.0f, u21 = 0.0f, u22 = 1.0f;
            int iters = 0;
            // HACK: call private function onAffineShapeFound even though we
            // are directly setting the shape to be circular.
            this->onAffineShapeFound(blur, x, y, s, pixelDistance, u11, u12, u21, u22, type, response, iters); // Call Step 4
        }
    }

    void extractDesc(int nKpts, float* kpts, uint8* desc)
    {
        /*
         The following are variable conventions in python.
         The code here might not be compliant.
         We should strive to change code HERE to
         match the python code, which is consistent

        Variables:
            invV : maps from ucircle onto an ellipse (perdoch.invA)
               V : maps from ellipse to ucircle      (perdoch.A)
               Z : the conic matrix                  (perdoch.E)
         */

        // Extract descriptors from user specified keypoints
        float x, y, iv11, iv12, iv21, iv22;
        float sc;
        float a11, a12, a21, a22, s, ori;
        for(int fx = 0; fx < nKpts; fx++)
        {
            // 2D Array offsets
            size_t rowk = fx * KPTS_DIM;
            size_t rowd = fx * DESC_DIM;
            //Read a keypoint from the file
            x = kpts[rowk + 0];
            y = kpts[rowk + 1];
            // We are currently using invV format in HotSpotter
            iv11 = kpts[rowk + 2];
            iv12 = 0;
            iv21 = kpts[rowk + 3];
            iv22 = kpts[rowk + 4];
            #if USE_ORI
            ori  = kpts[rowk + 5];
            #else
            ori  = R_GRAVITY_THETA
            #endif
            // Extract scale.
            sc = sqrt(std::abs((iv11 * iv22) - (iv12 * iv21)));
            // Deintegrate scale. Keep invA format
            s  = (sc / AffineShape::par.mrSize); // scale
            a11 = iv11 / sc;
            a12 = 0;
            a21 = iv21 / sc;
            a22 = iv22 / sc;
            // now sample the patch (populates this->patch)
            if(!this->normalizeAffine(this->image, x, y, s, a11, a12, a21, a22, ori))  //affine.cpp
            {
                this->populateDescriptor(desc, (fx * DESC_DIM)); // populate numpy array
            }
            else
            {
                printDBG("Failure!");
            }
        }
    }

    void extractPatches(int nKpts, float* kpts, float* patches_array)
    {
        /*
         * Warps underlying keypoints into patches
         */

        // Extract descriptors from user specified keypoints
        float x, y, iv11, iv12, iv21, iv22;
        float sc;
        float a11, a12, a21, a22, s, ori;
        for(int fx = 0; fx < nKpts; fx++)
        {
            // 2D Array offsets
            const size_t rowk = fx * KPTS_DIM;
            const size_t rowd = fx * DESC_DIM;
            //Read a keypoint from the file
            x = kpts[rowk + 0];
            y = kpts[rowk + 1];
            // We are currently using invV format in HotSpotter
            iv11 = kpts[rowk + 2];
            iv12 = 0;
            iv21 = kpts[rowk + 3];
            iv22 = kpts[rowk + 4];
            #if USE_ORI
            ori  = kpts[rowk + 5];
            #else
            ori  = R_GRAVITY_THETA
            #endif
            // Extract scale.
            sc = sqrt(std::abs((iv11 * iv22) - (iv12 * iv21)));
            // Deintegrate scale. Keep invA format
            s  = (sc / AffineShape::par.mrSize); // scale
            a11 = iv11 / sc;
            a12 = 0;
            a21 = iv21 / sc;
            a22 = iv22 / sc;

            int patch_size = AffineShape::par.patchSize;
            int patch_area = patch_size * patch_size;
            // now sample the patch (populates this->patch)
            if(!this->normalizeAffine(this->image, x, y, s, a11, a12, a21, a22, ori))  //affine.cpp
            {
                float* pp = this->patch.ptr<float>(0);
                int patch_size = this->patch.rows * this->patch.cols;
                for(int ix = 0; ix < patch_size; ix++)
                {
                    *patches_array = *pp;  // populate outvar
                    patches_array++;
                    pp++;
                }
            }
            else
            {
                // skip this data
                patches_array += patch_area;
                printDBG("Failure!");
            }
        }
    }

    //------------------------------------------------------------
    // BEGIN void onAffineShapeFound
    // *
    // * Callback for when an affine shape is found.
    // * This is the stack traceback for this function:
    // * {detectPyramidKeypoints ->
    // *  detectOctaveHessianKeypoints ->
    // *  localizeKeypoint ->
    // *  findAffineShape -> onAffineShapeFound}
    // * This function:
    // *   - Filters scales outside of bounds
    // *   - Computes the patch's rotation (if rotation_invariance is True)
    // *   - Computes the patch's SIFT Descriptor
    // *
    void onAffineShapeFound(const cv::Mat &blur, float x, float y,
                            float s, float pixelDistance,
                            float a11, float a12,
                            float a21, float a22,
                            int type, float response,
                            int iters)
    {
        /*
        Args:
            blur - the smoothed image at the current level of the gaussian pyramid
            x - x keypoint location
            y - y keypoint location
            s - keypoint scale (specifies determinant of shape matrix)
            a11, a12, a21, a22 - shape matrix (force to have determinant 1)
            type - can be one of {HESSIAN_DARK = 0, HESSIAN_BRIGHT = 1, HESSIAN_SADDLE = 2,}
            response - hessian responce
            iters - num iterations for shape estimation

        */
        // check if detected keypoint is within scale thresholds
        const float scale_min = hesPar.scale_min;
        const float scale_max = hesPar.scale_max;
        const float scale = s * AffineShape::par.mrSize;
        // negative thresholds turn the threshold test off
        if((scale_min > 0 && scale < scale_min) || (scale_max > 0 && scale > scale_max))
        {
            // failed scale threshold
            //printDBG("[shape_found] Shape Found And Failed")
            //printDBG("[shape_found]  * failed: " << scale)
            //printDBG("[shape_found]  * scale_min: " << scale_min << "; scale_max: " << scale_max)
            return;
        }
        //else
        //{
        //    //printDBG("[shape_found] Shape Found And Passed")
        //    //printDBG("[shape_found]  * passed: " << scale)
        //    //printDBG("[shape_found]  * scale_min: " << scale_min << "; scale_max: " << scale_max)
        //}
        // Enforce the gravity vector: convert shape into a up is up frame
        float ori = R_GRAVITY_THETA;
        rectifyAffineTransformationUpIsUp(a11, a12, a21, a22); // Helper
        std::vector<float> submaxima_oris;
        if(hesPar.rotation_invariance)
        {
            const bool passed = this->localizeKeypointOrientation(
                this->image, x, y, s, a11, a12, a21, a22, submaxima_oris);
            if (!passed)
            {
                return;
            }

            #define MAX_ORIS_PER_KEYPOINT 3
            if (submaxima_oris.size() > MAX_ORIS_PER_KEYPOINT)
            {
                return;
                //submaxima_oris.clear();
                //submaxima_oris.push_back(R_GRAVITY_THETA);
            }
            //submaxima_oris.push_back(2.0f);  // hack an orientations
            //submaxima_oris.push_back(R_GRAVITY_THETA);  // hack in a gravity orientation
        }
        else
        {
            // Just use one gravity orientation
            submaxima_oris.push_back(R_GRAVITY_THETA);
            if (hesPar.augment_orientation)
            {
                // +- 15 degrees or tau/24 ~= 0.26 radians
                submaxima_oris.push_back(R_GRAVITY_THETA + M_TAU / 24.0f);
                submaxima_oris.push_back(R_GRAVITY_THETA - M_TAU / 24.0f);
            }
        }
        //printDBG("[onAffShapeFound] Found " << submaxima_oris.size() << " orientations")
        global_c1++;
        global_nmulti_ori += static_cast<int>(submaxima_oris.size()) - 1;
        // push a keypoint for every orientation found
        for (int i = 0; i < submaxima_oris.size(); i++)
        {
            ori = submaxima_oris[i];
            global_nkpts++;
            // sample the patch (populates this->patch)
            // (from affine.cpp)
            //
            if (hesPar.only_count)
            {
                // HACK, if we are only counting we dont need to
                // interpolate new patches. (this does seem to cause
                // minor inconsistencies)
                if(!this->normalizeAffineCheckBorders(this->image, x, y, s, a11, a12, a21, a22, ori))
                {
                this->num_kpts++;
                }
            }
            else
            {
                if(!this->normalizeAffine(this->image, x, y, s, a11, a12, a21, a22, ori))
                {
                    this->push_new_keypoint(x, y, s, a11, a12, a21, a22, ori, type, response);
                }
            }
        }
        //else std::cout << global_nkpts << std::endl;
    }
    // END void onAffineShapeFound
    //------------------------------------------------------------
    void push_new_keypoint(float x, float y, float s, float a11, float a12,
                           float a21, float a22, float ori, int type,
                           float response)
    {
        this->num_kpts++;
        if (!hesPar.only_count)
        {
            //printDBG("response = " << response)
            // compute SIFT and append new keypoint and descriptor
            global_c1++;
            //this->DBG_patch();
            this->keys.push_back(Keypoint());
            Keypoint &k = this->keys.back();
            k.type = type;
            k.response = response;
            k.x = x;
            k.y = y;
            k.s = s;
            k.a11 = a11;
            k.a12 = a12;
            k.a21 = a21;
            k.a22 = a22;
            #if USE_ORI
            k.ori = ori;
            #endif
            this->populateDescriptor(k.desc, 0);
            //this->keys.push_back(Keypoint());
        }
    }

    float localizeKeypointOrientation(const cv::Mat& img, float x, float y,
            float s,
            float a11, float a12,
            float a21, float a22,
            std::vector<float>& submaxima_oris)
    {
        /*
        Finds keypoint orientation by warping keypoint into a unit circle, then
        computes orientation histogram with 36 bins using linear interpolation
        between bins, and fits a parabaloa (2nd degree taylor expansion)
        to localize sub-bin orientation.

        Args:
            img : an image
            pat : a keypoints image patch

        OutVars:
            submaxima_oris : dominant gradient orientations

        Returns:
            bool : success flag

        References:
             http://docs.opencv.org/modules/imgproc/doc/filtering.html?highlight=sobel#sobel

        CommandLine:
            python -m pyhesaff._pyhesaff --test-test_rot_invar --show --rebuild-hesaff --no-rmbuild
        */
        global_c2++;

        // Enforce that the shape is pointing down and sample the patch when the
        // orientation is the gravity vector
        const float ori = R_GRAVITY_THETA;
        rectifyAffineTransformationUpIsUp(a11, a12, a21, a22);
        // sample the patch (populates this->patch)
        if(this->normalizeAffine(img, x, y, s, a11, a12, a21, a22, ori))
        {
             // normalizerAffine is located in affine.cpp
             // normalizeAffine can fail if the keypoint is out of
             // bounds (consider adding an exception-based mechanism to
             // discard the keypoint?)
            return false;
        }
        // Warp elliptical keypoint region in image into a (cropped) unit circle
        //normalizeAffine does the job of ptool.get_warped_patch, but uses a
        //class variable to store the output (messy)
        // Compute gradient

        cv::Mat xgradient(this->patch.rows, this->patch.cols, this->patch.depth());
        cv::Mat ygradient(this->patch.rows, this->patch.cols, this->patch.depth());

        // NEW: COMPUTE GRADIENT WITH SOBEL
        cv::Sobel(this->patch, xgradient, this->patch.depth(),
                1, 0, 1, 1.0, 0, cv::BORDER_DEFAULT);
        cv::Sobel(this->patch, ygradient, this->patch.depth(),
                0, 1, 1, 1.0, 0, cv::BORDER_DEFAULT);

        // Compute magnitude and orientation
        cv::Mat orientations;
        cv::Mat magnitudes;
        //cv::magnitude(xgradient, ygradient, magnitudes);
        cv::cartToPolar(xgradient, ygradient, magnitudes, orientations);
        #if DEBUG_ROTINVAR
            this->DBG_dump_patch("orientationsbef", orientations);
        #endif

        orientations += M_GRAVITY_THETA; // adjust for 0 being downward
        // Keep orientations inside range (0, TAU)
        // Hacky, shoud just define modulus func for cvMat
        //orientations -= floor(orientations / M_TAU) * M_TAU

        for (int r = 0; r < orientations.rows; r+=1)
        {
            for (int c = 0; c < orientations.cols; c+=1)
            {
                if (orientations.at<float>(r, c) > M_TAU)
                {
                    orientations.at<float>(r, c) = orientations.at<float>(r, c) - M_TAU;
                }
                else if (orientations.at<float>(r, c) < 0)
                {
                    orientations.at<float>(r, c) = M_TAU + orientations.at<float>(r, c);
                }
            }
        }
        #if DEBUG_ROTINVAR
            this->DBG_dump_patch("orientationsaft", orientations);
        #endif

        // gaussian weight magnitudes
        //float sigma0 = (magnitudes.rows / 2) * .95;
        //float sigma1 = (magnitudes.cols / 2) * .95;
        const float sigma0 = (static_cast<float>(magnitudes.rows) / 2.0f) * .4f;
        const float sigma1 = (static_cast<float>(magnitudes.cols) / 2.0f) * .4f;
        cv::Mat gauss_weights;
        make_2d_gauss_patch_01(magnitudes.rows, magnitudes.cols, sigma0,
                sigma1, gauss_weights);
        // Weight magnitudes using a gaussian kernel
        cv::Mat weights = magnitudes.mul(gauss_weights);

        #if DEBUG_ROTINVAR
            //this->DBG_kp_shape_a(x, y, s, a11, a12, a21, a22, ori);
            //this->DBG_print_mat(this->patch, 10, "PATCH");
            //this->DBG_dump_patch("PATCH", this->patch);
            //this->DBG_print_mat(xgradient, 10, "xgradient");
            //this->DBG_print_mat(ygradient, 10, "ygradient");
            std::string patch_fpath = this->DBG_dump_patch("PATCH", this->patch);
            std::string gradx_fpath = this->DBG_dump_patch("xgradient", xgradient);
            std::string grady_fpath = this->DBG_dump_patch("ygradient", ygradient);
            //printDBG("d0_max = " << d0_max);
            //printDBG("d1_max = " << d1_max);
            //this->DBG_print_mat(gauss_kernel_d0, 1, "GAUSSd0");
            //this->DBG_print_mat(gauss_kernel_d1, 1, "GAUSSd1");
            //this->DBG_print_mat(gauss_weights, 10, "GAUSS");
            //this->DBG_print_mat(orientations, 10, "ORI");
            std::string gmag_fpath = this->DBG_dump_patch("magnitudes", magnitudes);
            std::string gaussweight_fpath = this->DBG_dump_patch("gauss_weights", gauss_weights, true);
            //std::str ori_fpath;
            //std::str weights_fpath;

            std::string weights_fpath = this->DBG_dump_patch("WEIGHTS", weights);
            this->DBG_dump_patch("orientations", orientations);
            cv::Mat orientations01 = orientations.mul(255.0 / 6.28);
            std::string ori_fpath01 = this->DBG_dump_patch("orientations01", orientations01);
            cv::Mat gaussweight01 = gauss_weights.mul(255.0 / 6.28);
            std::string gaussweight01_fpath = this->DBG_dump_patch("gaussweight01", gaussweight01);
            //make_str(ori_fpath, "patches/KP_" << this->keys.size() << "_" << "orientations01" << ".png");
            //make_str(weights_fpath, "patches/KP_" << this->keys.size() << "_" << "WEIGHTS" << ".png");
            //this->DBG_print_mat(magnitudes, 10, "MAG");
            //this->DBG_print_mat(weights, 10, "WEIGHTS");
            //cv::waitKey(0);
        #endif

        // HISTOGRAM INTERPOLOATION PART
        // Compute ori histogram, splitting votes using linear interpolation
        const int nbins = 36;
        Histogram<float> hist = computeInterpolatedHistogram<float>(
                orientations.begin<float>(), orientations.end<float>(),
                weights.begin<float>(), weights.end<float>(),
                nbins, M_TAU, 0.0f);

        Histogram<float> wrapped_hist = htool::wrap_histogram(hist);
        std::vector<float> submaxima_xs, submaxima_ys;
        // inplace wrap histogram (because orientations are circular)
        htool::hist_edges_to_centers(wrapped_hist);
        // Compute orientation as maxima of wrapped histogram
        const float maxima_thresh = this->hesPar.ori_maxima_thresh;

        htool::argsubmaxima(
                wrapped_hist, submaxima_xs, submaxima_ys, maxima_thresh);
        for (int i = 0; i < submaxima_xs.size(); i ++ )
        {
            float submax_ori = submaxima_xs[i];
            float submax_ori2 = ensure_0toTau<float>(submax_ori);
            submaxima_oris.push_back(submax_ori);
            #if DEBUG_ROTINVAR
                if (submax_ori != submax_ori2)
                {
                    printDBG("[find_ori] +------------")
                    printDBG("[find_ori] submax_ori  = " << submax_ori)
                    printDBG("[find_ori] submax_ori2 = " << submax_ori2)
                    printDBG("[find_ori] L____________")
                }
            #endif
            //submax_ori = submax_ori2;
            //submaxima_oris.push_back(submax_ori);
        }

        #if DEBUG_ROTINVAR
            /*
             python -m pyhesaff._pyhesaff --test-test_rot_invar --show --rebuild-hesaff --no-rmbuild
             python -m pyhesaff._pyhesaff --test-test_rot_invar --show
             */

            //make_str(cmd_str1,
            //        "python -m vtool.patch --test-test_ondisk_find_patch_fpath_dominant_orientations --show" <<
            //        " --patch-fpath " << patch_fpath <<
            //        "&"
            //        ""
            //        );
            //run_system_command(cmd_str1);

            make_str(cmd_str2,
                    "python -m vtool.histogram --test-show_ori_image_ondisk --show" <<
                    " --patch_img_fpath "   << patch_fpath <<
                    " --ori_img_fpath "     << ori_fpath01 <<
                    " --weights_img_fpath " << weights_fpath <<
                    " --grady_img_fpath "   << grady_fpath <<
                    " --gradx_img_fpath "   << gradx_fpath <<
                    " --gauss_weights_img_fpath "   << gaussweight01_fpath <<
                    " --title cpp_show_ori_ondisk "
                    "&"
                    );
            run_system_command(cmd_str2);

            print_vector<float>(wrapped_hist.data, "wrapped_hist");
            print_vector<float>(wrapped_hist.edges, "wrapped_edges");
            print_vector<float>(wrapped_hist.centers, "wrapped_centers");

            show_hist_submaxima(wrapped_hist);
        #endif
        return true;
    }


    void populateDescriptor(uint8* desc, size_t offst)
    {
        this->sift.computeSiftDescriptor(this->patch);
        for(int ix = 0; ix < DESC_DIM; ix++)
        {
            desc[offst + ix] = (uint8) sift.vec[ix];  // populate outvar
        }
    }

    std::string DBG_dump_patch(std::string str_name, cv::Mat& dbgpatch, bool fix=false)
    {
        /*
        CommandLine:
            ./unix_build.sh --fast && ./build/hesaffexe /home/joncrall/.config/utool/star.png
            ./unix_build.sh --fast && ./build/hesaffexe /home/joncrall/.config/utool/star.png
            sh mingw_build.sh --fast
            build/hesaffexe /home/joncrall/.config/utool/star.png

            python -c "import utool; utool.cmd('build/hesaffexe.exe ' + utool.grab_test_imgpath('star.png'))"

        References:
            http://docs.opencv.org/modules/core/doc/basic_structures.html#mat-depth
            http://stackoverflow.com/questions/23019021/opencv-how-to-save-float-array-as-an-image

            define CV_8U   0 define CV_8S   1 define CV_16U  2 define CV_16S  3
            define CV_32S  4 define CV_32F  5 define CV_64F  6

             cvk = 'CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F'.split()
             print(ut.repr2(dict(zip(cvk, ut.dict_take(cv2.__dict__, cvk)))))
         */
        //DBG: write out patches
        run_system_command("python -c \"import utool as ut; ut.ensuredir('patches', verbose=False)\"");
        make_str(patch_fpath, "patches/KP_" << this->keys.size() << "_" << str_name << ".png");
        printDBG("[DBG] ----------------------")
        printDBG("[DBG] Dumping patch to patch_fpath = " << patch_fpath);
        printDBG("[DBG] patch.shape = (" <<
                dbgpatch.rows << ", " <<
                dbgpatch.cols << ", " <<
                dbgpatch.channels() << ")"
                )
        printDBG("[DBG] patch.type() = " <<  dbgpatch.type());
        printDBG("[DBG] patch.depth() = " <<  dbgpatch.depth());
        printDBG("[DBG] patch.at(0, 0) = " <<  (dbgpatch.at<float>(0, 0)));
        //printDBG(this->patch)
        cv::Mat dbgpatch_;
        if (fix)
        {
            cv::Mat dbgpatch_fix = dbgpatch.mul(255.0);
            //cv::Mat dbgpatch_fix = dbgpatch;
            dbgpatch_fix.convertTo(dbgpatch_, CV_8U);
        }
        else
        {
            dbgpatch.convertTo(dbgpatch_, CV_8U);
        }
        //cv::namedWindow(patch_fpath, cv::WINDOW_NORMAL);
        //cv::imshow(patch_fpath, dbgpatch_);
        //cv::waitKey(0);
        cv::imwrite(patch_fpath, dbgpatch);
        return patch_fpath;
    }

    void DBG_keypoint(float* kpts, int rowk)
    {
        float x, y, iv11, iv12, iv21, iv22, ori;
        x = kpts[rowk + 0];
        y = kpts[rowk + 1];
        // We are currently using invV format in HotSpotter
        iv11 = kpts[rowk + 2];
        iv12 = 0;
        iv21 = kpts[rowk + 3];
        iv22 = kpts[rowk + 4];
        ori  = kpts[rowk + 5];
        DBG_kp_shape_invV(x, y, iv11, iv12, iv21, iv22, ori);
    }

    void DBG_kp_shape_invV(float x, float y, float iv11, float iv12,
                     float iv21, float iv22, float ori)
    {
        printDBG("+---");
        printDBG("|   xy = (" <<  x << ", " <<  y << ") ");
        printDBG("| invV = [(" << iv11 << ", " << iv12 << "), ");
        printDBG("|         (" << iv21 << ", " << iv22 << ")] ");
        printDBG("|  ori = " << ori);
        printDBG("L___");
    }

    void DBG_kp_shape_a(float x, float y, float s, float a11, float a12,
                        float a21, float a22, float ori)
    {
        float sc = s * AffineShape::par.mrSize;
        printDBG("+---");
        printDBG("|        xy = (" <<  x << ", " <<  y << ") ");
        printDBG("| hat{invV} = [(" << a11 << ", " << a12 << "), ");
        printDBG("|           (" << a21 << ", " << a22 << ")] ");
        printDBG("|         sc = "   << sc);
        printDBG("|          s = "   << s);
        printDBG("|        ori = " << ori);
        printDBG("L___");
    }

    void DBG_print_mat(Mat& M, int stride=1, const char* name="M")
    {
        printDBG("----------------------")
        printDBG("Matrix Info")
        printDBG(name << ".shape = (" <<
                M.rows << ", " <<
                M.cols << ", " <<
                M.channels() << ")"
                )
        printDBG(name << ".type() = " <<  M.type());
        printDBG(name << ".depth() = " <<  M.depth());
        printDBG(name << ".at(0, 0) = " <<  (M.at<float>(0, 0)));
        int channels = M.channels();
        std::cout << name << " = cv::Matrix[\n";
        for (int r = 0; r < M.rows; r+=stride)
        {
            std::cout << "    [";
            for (int c = 0; c < M.cols; c+=stride)
            {
                if (M.type() == 5)
                {
                std::cout <<
                    //std::fixed << std::setw(5) << std::setprecision(2) <<
                    M.at<float>(r, c) << ", ";
                }
                else if (M.type() == 6)
                {
                std::cout <<
                    //std::fixed << std::setw(5) << std::setprecision(2) <<
                    M.at<double>(r, c) << ", ";
                }
                //std::cout << "[";
                //for (int d=0; d < channels; d++)
                //{
                //}
                //std::cout << "]";
            }
            std::cout << "]" << std::endl;
        }
        std::cout << "]" << std::endl;
    }

    void DBG_params()
    {
    //printDBG("pyrParams.numberOfScales      = " << pyrParams.numberOfScales);
    //printDBG("pyrParams.threshold           = " << pyrParams.threshold);
    //printDBG("pyrParams.edgeEigenValueRatio = " << pyrParams.edgeEigenValueRatio);
    //printDBG("pyrParams.border              = " << pyrParams.border);
    //printDBG("pyrParams.maxPyramidLevels    = " << pyrParams.maxPyramidLevels);
    //printDBG("pyrParams.initialSigma        = " << pyrParams.initialSigma);
    //printDBG("affShapeParams.maxIterations        = " << affShapeParams.maxIterations);
    //printDBG("affShapeParams.convergenceThreshold = " << affShapeParams.convergenceThreshold);
    //printDBG("affShapeParams.smmWindowSize        = " << affShapeParams.smmWindowSize);
    //printDBG("affShapeParams.mrSize               = " << affShapeParams.mrSize);
    //printDBG("affShapeParams.initialSigma         = " << affShapeParams.initialSigma);
    //printDBG("affShapeParams.patchSize            = " << affShapeParams.patchSize);
    //printDBG("siftParams.spatialBins     = " << siftParams.spatialBins);
    //printDBG("siftParams.orientationBins = " << siftParams.orientationBins);
    //printDBG("siftParams.maxBinValue     = " << siftParams.maxBinValue);
    //printDBG("siftParams.patchSize       = " << siftParams.patchSize);
    printDBG(" * hesPar.scale_min            = " << hesPar.scale_min);
    printDBG(" * hesPar.scale_max            = " << hesPar.scale_max);
    printDBG(" * hesPar.rotation_invariance  = " << hesPar.rotation_invariance);
    printDBG(" * hesPar.augment_orientation  = " << hesPar.augment_orientation);
    printDBG(" * hesPar.ori_maxima_thresh    = " << hesPar.ori_maxima_thresh);
    printDBG(" * hesPar.affine_invariance    = " << hesPar.affine_invariance);
    }


};
// END class AffineHessianDetector



//----------------------------------------------
// BEGIN PYTHON BINDINGS
// * python's ctypes module can talk to extern c code
// http://nbviewer.ipython.org/github/pv/SciPy-CookBook/blob/master/ipython/Ctypes.ipynb
#ifdef __cplusplus
extern "C" {
#endif

// Python binds to extern C code
//#define HESAFF_EXPORTED extern HESAFF_EXPORTED


HESAFF_EXPORTED int detect(AffineHessianDetector* detector)
{
    printDBG("detector->detect");
    int nKpts = detector->detect();
    printDBG("[detect] nKpts = " << nKpts);
    printDBG("[detect] global_c1 = " << global_c1);
    printDBG("[detect] global_c2 = " << global_c2);
    printDBG("[detect] global_nkpts = " << global_nkpts);
    printDBG("[detect] global_nmulti_ori = " << global_nmulti_ori);
    return nKpts;
}


HESAFF_EXPORTED int get_cpp_version()
{
    return 4;
}


HESAFF_EXPORTED int is_debug_mode()
{
    return DEBUG_ROTINVAR || DEBUG_HESAFF;
}


HESAFF_EXPORTED int get_kpts_dim()
{
    return KPTS_DIM;
}

HESAFF_EXPORTED int get_desc_dim()
{
    return DESC_DIM;
}


// MACROS TO REDUCE REDUNDANT FUNCTION SIGNATURE ARGUMENTS

// Macro for putting arguments into the call signature
#define __HESAFF_PARAM_SIGNATURE_ARGS__ \
 int   numberOfScales,          \
 float threshold,               \
 float edgeEigenValueRatio,     \
 int   border,                  \
 int   maxPyramidLevels,        \
 int   maxIterations,           \
 float convergenceThreshold,    \
 int   smmWindowSize,           \
 float mrSize,                  \
 int   spatialBins,             \
 int   orientationBins,         \
 float maxBinValue,             \
 float initialSigma,            \
 int   patchSize,               \
 float scale_min,               \
 float scale_max,               \
 bool  rotation_invariance,     \
 bool  augment_orientation,     \
 float ori_maxima_thresh,       \
 bool  affine_invariance,       \
 bool  only_count,              \
 bool  use_dense,               \
 int   dense_stride,            \
 float siftPower


// Macro for putting calling a function with the macroed signature
#define __HESAFF_PARAM_CALL_ARGS__                                                       \
numberOfScales, threshold, edgeEigenValueRatio, border, maxPyramidLevels, maxIterations, \
convergenceThreshold, smmWindowSize, mrSize, spatialBins, orientationBins,               \
maxBinValue, initialSigma, patchSize, scale_min, scale_max,                              \
rotation_invariance, augment_orientation, ori_maxima_thresh,                             \
affine_invariance, only_count, use_dense, dense_stride, siftPower


#define __MACRO_COMMENT__(s) ;

// Macro to define the param object in func with call signature
#define __HESAFF_DEFINE_PARAMS_FROM_CALL__                      \
    __MACRO_COMMENT__( Define params)                           \
    SIFTDescriptorParams siftParams;                            \
    PyramidParams pyrParams;                                    \
    AffineShapeParams affShapeParams;                           \
    HesaffParams hesParams;                                     \
                                                                \
    __MACRO_COMMENT__( Copy Pyramid params)                     \
    pyrParams.numberOfScales            = numberOfScales;       \
    pyrParams.threshold                 = threshold;            \
    pyrParams.edgeEigenValueRatio       = edgeEigenValueRatio;  \
    pyrParams.border                    = border;               \
    pyrParams.maxPyramidLevels          = maxPyramidLevels;     \
    pyrParams.initialSigma              = initialSigma;         \
                                                                \
    __MACRO_COMMENT__( Copy Affine Shape params)                \
    affShapeParams.maxIterations        = maxIterations;        \
    affShapeParams.convergenceThreshold = convergenceThreshold; \
    affShapeParams.smmWindowSize        = smmWindowSize;        \
    affShapeParams.mrSize               = mrSize;               \
    affShapeParams.initialSigma         = initialSigma;         \
    affShapeParams.patchSize            = patchSize;            \
                                                                \
    __MACRO_COMMENT__( Copy SIFT params)                        \
    siftParams.spatialBins              = spatialBins;          \
    siftParams.orientationBins          = orientationBins;      \
    siftParams.maxBinValue              = maxBinValue;          \
    siftParams.patchSize                = patchSize;            \
                                                                \
    __MACRO_COMMENT__( Copy my params)                          \
    hesParams.scale_min            = scale_min;                 \
    hesParams.scale_max            = scale_max;                 \
    hesParams.rotation_invariance  = rotation_invariance;       \
    hesParams.augment_orientation  = augment_orientation;       \
    hesParams.ori_maxima_thresh    = ori_maxima_thresh;         \
    hesParams.affine_invariance    = affine_invariance;         \
    hesParams.only_count           = only_count;                \
    __MACRO_COMMENT__( )                                        \
    pyrParams.use_dense                 = use_dense;            \
    pyrParams.dense_stride              = dense_stride;         \
    siftParams.siftPower                = siftPower;

// Macro to define the param object in func without call signature
#define __HESAFF_DEFINE_PARAMS_FROM_DEFAULTS__  \
    __MACRO_COMMENT__( Pyramid Params)          \
    const int   numberOfScales = 3;             \
    const float threshold = 16.0f / 3.0f;       \
    const float edgeEigenValueRatio = 10.0f;    \
    const int   border = 5;                     \
    const int   maxPyramidLevels = -1;          \
    __MACRO_COMMENT__( Affine Params Shape)     \
    const int   maxIterations = 16;             \
    const float convergenceThreshold = 0.05f;   \
    const int   smmWindowSize = 19;             \
    const float mrSize = 3.0f * sqrt(3.0f);     \
    __MACRO_COMMENT__( SIFT params)             \
    const int spatialBins = 4;                  \
    const int orientationBins = 8;              \
    const float maxBinValue = 0.2f;             \
    __MACRO_COMMENT__( Shared Pyramid + Affine) \
    const float initialSigma = 1.6f;            \
    __MACRO_COMMENT__( Shared SIFT + Affine)    \
    const int patchSize = 41;                   \
    __MACRO_COMMENT__( My params)               \
    const float scale_min = -1.0f;              \
    const float scale_max = -1.0f;              \
    const bool rotation_invariance = false;     \
    const bool augment_orientation = false;     \
    const float ori_maxima_thresh = .8f;        \
    const bool affine_invariance = true;        \
    __MACRO_COMMENT__( )                        \
    const bool use_dense = false;               \
    const int  dense_stride = 32;               \
    const float siftPower = 1.0f;               \
    const bool only_count = false;


// new hessian affine detector (from image pixels)
HESAFF_EXPORTED AffineHessianDetector* new_hesaff_image(uint8 *imgin, int rows, int cols, int  channels, __HESAFF_PARAM_SIGNATURE_ARGS__)
{
    // Convert input image to float32
    cv::Mat image(rows, cols, CV_32FC1, Scalar(0));
    float *imgout = image.ptr<float>(0);
    if (channels == 3)
    {
        for(size_t i = rows * cols; i > 0; i--)
        {
            *imgout = (float(imgin[0]) + imgin[1] + imgin[2]) / 3.0f;
            imgout++;
            imgin += 3;
        }
    }
    else if (channels == 1)
    {
        for(size_t i = rows * cols; i > 0; i--)
        {
            *imgout = float(imgin[0]);
            imgout++;
            imgin += 1;
        }
    }

    __HESAFF_DEFINE_PARAMS_FROM_CALL__

    // Create detector
    AffineHessianDetector* detector = new AffineHessianDetector(image, pyrParams, affShapeParams, siftParams, hesParams);
    detector->DBG_params();
    return detector;
}

// new hessian affine detector (from image fpath)
HESAFF_EXPORTED AffineHessianDetector* new_hesaff_fpath(char* img_fpath, __HESAFF_PARAM_SIGNATURE_ARGS__)
{
    printDBG("making detector for " << img_fpath);
    printDBG(" * img_fpath = " << img_fpath);
    // Read in image
    cv::Mat tmp = cv::imread(img_fpath);
    const int rows = tmp.rows;
    const int cols = tmp.cols;
    const int channels = 3;
    uint8 *imgin  = tmp.ptr<uint8>(0);
    // Create detector
    AffineHessianDetector* detector = new_hesaff_image(imgin, rows, cols, channels, __HESAFF_PARAM_CALL_ARGS__);
    return detector;
}


// new default hessian affine detector WRAPPER
HESAFF_EXPORTED AffineHessianDetector* new_hesaff_imgpath_noparams(char* img_fpath)
{

    __HESAFF_DEFINE_PARAMS_FROM_DEFAULTS__

    AffineHessianDetector* detector = new_hesaff_fpath(img_fpath, __HESAFF_PARAM_CALL_ARGS__);
    return detector;
}

HESAFF_EXPORTED void free_hesaff(AffineHessianDetector* detector)
{
    printDBG("about to free detector=@" << static_cast<void*>(detector))
    //printDBG("about to free &detector=@" << static_cast<void*>(&detector))
    delete detector;
    printDBG("deallocated detector");
}

// extract descriptors from user specified keypoints
HESAFF_EXPORTED void extractDesc(AffineHessianDetector* detector,
                          int nKpts, float* kpts, uint8* desc)
{
    printDBG("detector->extractDesc");
    detector->extractDesc(nKpts, kpts, desc);
    printDBG("extracted nKpts = " << nKpts);
}


// Extracts patches used to compute descriptors
HESAFF_EXPORTED void extractPatches(AffineHessianDetector* detector,
                          int nKpts, float* kpts, float* patch_array)
{
    printDBG("detector->extractPatches");
    detector->extractPatches(nKpts, kpts, patch_array);
    printDBG("extracted nKpts = " << nKpts);
}

// export current detections to numpy arrays
HESAFF_EXPORTED void exportArrays(AffineHessianDetector* detector,
                           int nKpts, float *kpts, uint8 *desc)
{
    printDBG("detector->exportArrays(" << nKpts << ")");
    //printDBG("detector->exportArrays kpts[0]" << kpts[0] << ")");
    //printDBG("detector->exportArrays desc[0]" << (int) desc[0] << ")");
    detector->exportArrays(nKpts, kpts, desc);
    //printDBG("detector->exportArrays kpts[0]" << kpts[0] << ")");
    //printDBG("detector->exportArrays desc[0]" << (int) desc[0] << ")");
    printDBG("FINISHED detector->exportArrays");
}

// dump current detections to disk
HESAFF_EXPORTED void writeFeatures(AffineHessianDetector* detector,
                            char* img_fpath)
{
    // Dump keypoints to disk in text format
    char suffix[] = ".hesaff.sift";
    const int len = static_cast<int>(strlen(img_fpath) + strlen(suffix)) + 1;

    #ifdef WIN32
      // windows doesnt follow the ISO C99 snprintf standard.
      // https://stackoverflow.com/questions/1270387/are-snprintf-and-friends-safe-to-use
      char* out_fpath = new char[len];
      snprintf_s(out_fpath, len, len, "%s%s", img_fpath, suffix);
    #else
      char out_fpath[len];
      snprintf(out_fpath, len, "%s%s", img_fpath, suffix);
    #endif

    out_fpath[len - 1] = 0;
    printDBG("detector->writing_features: " << out_fpath);
    std::ofstream out(out_fpath);
    detector->exportKeypoints(out);
    // Clean Up
    #ifdef WIN32
    delete[] out_fpath;
    #endif
}

HESAFF_EXPORTED void extractDescFromPatches(int num_patches,
                                     int patch_h,
                                     int patch_w,
                                     uint8* patches_array,
                                     uint8* descriptors_array)
{
    // Function to extract SIFT descriptors from an array of patches
    // TODO: paramatarize
    SIFTDescriptorParams siftParams;
    siftParams.patchSize = patch_h;

    SIFTDescriptor sift(siftParams);

    //cv::Mat patch(patch_w, patch_h, CV_8U);
    cv::Mat patch(patch_w, patch_h, CV_32F);

    printDBG("num_patches=" << num_patches);
    printDBG("patch_h=" << patch_h);
    printDBG("patch_w=" << patch_w);
    //printf("patches_array[-16]=%016x\n", patches_array[-16]);
    float *pp;

    for(int i = 0; i < num_patches; i++)
    {
        pp = patch.ptr<float>(0);
        for(int r = 0; r < patch_h; r++)
        {
            for(int c = 0; c < patch_w; c++)
            {
                *pp = (float) patches_array[(i * patch_h * patch_w) + (r * patch_w) + c];
                pp++;
            }
        }
        sift.computeSiftDescriptor(patch);
        for(int ix = 0; ix < DESC_DIM; ix++)
        {
            // populate outvar
            descriptors_array[(i * DESC_DIM) + ix] = (uint8) sift.vec[ix];
        }
    }

}


HESAFF_EXPORTED AffineHessianDetector** detectFeaturesListStep1(int num_fpaths,
                                                          char** image_fpath_list,
                                                          __HESAFF_PARAM_SIGNATURE_ARGS__)
{
    printDBG("detectFeaturesListStep1()");
    // Create all of the detector_array
    AffineHessianDetector** detector_array = new AffineHessianDetector*[num_fpaths];
    int index;
    //#pragma omp parallel for private(index)
    for(index = 0; index < num_fpaths; ++index)
    {
        char* image_filename = image_fpath_list[index];
        AffineHessianDetector* detector = new_hesaff_fpath(image_filename, __HESAFF_PARAM_CALL_ARGS__);
        //detector->DBG_params();
        detector_array[index] = detector;
    }
    return detector_array;
}

HESAFF_EXPORTED void detectFeaturesListStep2(int num_fpaths, AffineHessianDetector** detector_array, int* length_array)
{
    printDBG("detectFeaturesListStep2()");
    // Run Detection
    int index;
    //#pragma omp parallel for private(index)
    for(index = 0; index < num_fpaths; ++index)
    {
        AffineHessianDetector* detector = detector_array[index];
        int length = detector->detect();
        length_array[index] = length;
    }
}

HESAFF_EXPORTED void detectFeaturesListStep3(int num_fpaths,
                                       AffineHessianDetector** detector_array,
                                       int* length_array,
                                       int* offset_array,
                                       float* flat_keypoints,
                                       uint8* flat_descriptors)
{
    printDBG("detectFeaturesListStep3()");
    // Export the results
    int index;
    //#pragma omp parallel for private(index)
    for(index = 0; index < num_fpaths; ++index)
    {
        AffineHessianDetector* detector = detector_array[index];
        int length = length_array[index];
        int offset = offset_array[index];
        printDBG("offset " << offset)
        printDBG("length " << length)
        float *keypoints = &flat_keypoints[offset * KPTS_DIM];
        uint8 *descriptors = &flat_descriptors[offset * DESC_DIM];
        exportArrays(detector, length, keypoints, descriptors);
    }
    // Clean up hesaff objects
    for(index = 0; index < num_fpaths; ++index)
    {
        delete detector_array[index];
    }
    delete detector_array;
}
}


int main(int argc, char **argv)
{
    /*
    program entry point for command line use if we build the executable

    CommandLine:
         ./unix_build.sh --fast && ./build/hesaffexe /home/joncrall/.config/utool/star.png
         ./unix_build.sh --fast && ./build/hesaffexe /home/joncrall/.config/utool/star.png
         sh mingw_build.sh --fast
         ~/code/hesaff/build/hesaffexe ~/.config/utool/star.png -rotation_invariance
         ~/code/hesaff/build/hesaffexe ~/.config/utool/lena.png -rotation_invariance

         hes
         cd build
         cp /home/joncrall/.config/utool/lena.png .

         hes
         cd build
         ./hesaffexe lena.png

         gprof hesaffexe
         gprof hesaffexe | gprof2dot | dot -Tpng -o output.png
         eog output.png
    */
    const char* about_message = "\nUsage: hesaffexe image_name.png\nDescribes elliptical keypoints (with gravity vector) given in kpts_file.txt using a SIFT descriptor. The help message has unfortunately been deleted. Check github history for details. https://github.com/perdoch/hesaff/blob/master/hesaff.cpp\n\n";
    // Parser Reference: http://docs.opencv.org/trunk/modules/core/doc/command_line_parser.html

    if(argc > 1)
    {
        printDBG("main()");
        char* img_fpath = argv[1];
        int nKpts;
        AffineHessianDetector* detector = new_hesaff_imgpath_noparams(img_fpath);
        //detector->hesPar.rotation_invariance = true;
            //(argc > 2) ? atoi(argv[2]) : true;
        detector->DBG_params();
        nKpts = detect(detector);
        writeFeatures(detector, img_fpath);
        std::cout << "[main] nKpts: " << nKpts << std::endl;
        std::cout << "[main] nKpts_: " << detector->keys.size() << std::endl;
        std::cout << "[main] global_nkpts: " << global_nkpts << std::endl;
        std::cout << "[main] global_c1: " << global_c1 << std::endl;
        std::cout << "[main] global_c2: " << global_c2 << std::endl;
        delete detector;
    }
    else
    {
        printf("%s", about_message);
    }
}
