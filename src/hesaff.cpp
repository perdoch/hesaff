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
/*
CommandLine:
    mingw_build.bat && python -c "import utool as ut; ut.cmd('build/hesaffexe.exe ' + ut.grab_test_imgpath('star.png'))"
    ./unix_build.sh && python -c "import utool as ut; ut.cmd('build/hesaffexe ' + ut.grab_test_imgpath('star.png'))"

    python -m pyhesaff._pyhesaff --test-test_rot_invar --show --rebuild-hesaff --no-rmbuild
    python -m pyhesaff._pyhesaff --test-test_rot_invar --show


 */

// Main File. Includes and uses the other files
//

#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/core/core.hpp>
//#include <opencv2/core/utility.hpp>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

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
#define M_PI 3.14159
#endif

#ifndef M_TAU
#define M_TAU 6.28318
#endif

// Gravity points downward = tau / 4 = pi / 2
#ifndef M_GRAVITY_THETA
#define M_GRAVITY_THETA 1.570795
// relative to gravity
#define R_GRAVITY_THETA 0
#endif

#define DEBUG_HESAFF 0

#if DEBUG_HESAFF
    #define printDBG(msg) std::cout << "[hesaff.c] " << msg << std::endl;
    #define write(msg) std::cout << msg;
#else
    #define printDBG(msg);
#endif


#define USE_ORI 1  // developing rotational invariance


#if USE_ORI
const int KPTS_DIM = 6;
#else
const int KPTS_DIM = 5;
#endif
const int DESC_DIM = 128;

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
    HesaffParams hesPar;
public:
    // Constructor
    AffineHessianDetector(const cv::Mat &image,
                          const PyramidParams &par,
                          const AffineShapeParams &ap,
                          const SIFTDescriptorParams &sp,
                          const HesaffParams& hesParams):
        HessianDetector(par), AffineShape(ap), image(image), sift(sp), hesPar(hesParams)
    {
        
        this->setHessianKeypointCallback(this); //Inherits from pyramid.h HessianDetector
        this->setAffineShapeCallback(this); // Inherits from affine.h AffineShape
    }

    int detect()
    {
        // Reset counters
        this->detectPyramidKeypoints(this->image);
        return this->keys.size();
    }

    void exportArrays(int nKpts, float *kpts, uint8 *desc)
    {
        // Exports keypoints and descriptors into preallocated numpy arrays
        for(size_t fx = 0; fx < nKpts; fx++)
        {
            Keypoint &k = keys[fx];
            float x, y, iv11, iv12, iv21, iv22, s, det;
            float sc = AffineShape::par.mrSize * k.s;
            size_t rowk = fx * KPTS_DIM;
            size_t rowd = fx * DESC_DIM;
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

            #if 0
            //if(fx == 0 || fx == nKpts - 1){
            //    DBG_keypoint(kpts, rowk);
            //}
            #endif
            // Assign Descriptor Output
            for(size_t ix = 0; ix < DESC_DIM; ix++)
            {
                desc[rowd + ix] = uint8(k.desc[ix]);
            }
        }
    }

    void write_features(char* img_fpath)
    {
        // Dump keypoints to disk in text format
        char suffix[] = ".hesaff.sift";
        int len = strlen(img_fpath) + strlen(suffix) + 1;
        #ifdef WIN32
        char* out_fpath = new char[len];
        #else
        char out_fpath[len];
        #endif
        snprintf(out_fpath, len, "%s%s", img_fpath, suffix);
        out_fpath[len - 1] = 0;
        printDBG("detector->writing_features: " << out_fpath);
        std::ofstream out(out_fpath);
        this->exportKeypoints(out);
        // Clean Up
        #ifdef WIN32
        delete[] out_fpath;
        #endif
    }

    void exportKeypoints(std::ostream &out)
    {
        /*Writes text keypoints in the invE format to a stdout stream
         * [iE_a, iE_b]
         * [iE_b, iE_d]
         */
        out << DESC_DIM << std::endl;
        int nKpts = keys.size();
        printDBG("[export] Writing " << nKpts << " keypoints");
        out << nKpts << std::endl;
        for(size_t i = 0; i < nKpts; i++)
        {
            Keypoint &k = keys[i];
            float sc = AffineShape::par.mrSize * k.s;
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
            float *diag_invA = (float *)svd_invA.w.data;
            // Integrate scale into 1/S and take squared inverst to make 1/W
            diag_invA[0] = 1.0f / (diag_invA[0] * diag_invA[0] * sc * sc);
            diag_invA[1] = 1.0f / (diag_invA[1] * diag_invA[1] * sc * sc);
            // Build the matrix invE
            // (I dont understand why U here, but it preserves the rotation I guess)
            // invE = (V * 1/S * U.T) * (U * 1/S * V.T)
            cv::Mat invE = svd_invA.u * cv::Mat::diag(svd_invA.w) * svd_invA.u.t();
            // Write inv(E) to out stream
            float e11 = invE.at<float>(0, 0);
            float e12 = invE.at<float>(0, 1); // also e12 because of E symetry
            float e22 = invE.at<float>(1, 1);
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
            // the callback is private to hack a call to onAffineShapeFound
            // directly
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
        for(int fx = 0; fx < (nKpts); fx++)
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

    //------------------------------------------------------------
    // BEGIN void onAffineShapeFound
    // *
    // * Callback for when an affine shape is found.
    // * This is the stack traceback for this function:
    // * {detectPyramidKeypoints ->
    // *  detectOctaveKeypoints ->
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
        // type can be one of:
        #if DEBUG_ROTINVAR 
        if (std::abs(response) > 1000 || std::abs(response) < 800)
        {
            return;
        }
        #endif

        // check if detected keypoint is within scale thresholds
        float scale_min = hesPar.scale_min;
        float scale_max = hesPar.scale_max;
        //float scale = AffineShape::par.mrSize * s;
        //float scale = s * AffineShape::par.mrSize / (AffineShape::par.initialSigma * pixelDistance);
        float scale = s * AffineShape::par.mrSize;
        // negative thresholds turn the threshold test off
        if((scale_min > 0 && scale < scale_min) || (scale_max > 0 && scale > scale_max))
        {
            // failed scale threshold
            //printDBG("[shape_found] Shape Found And Failed")
            //printDBG("[shape_found]  * failed: " << scale)
            //printDBG("[shape_found]  * scale_min: " << scale_min << "; scale_max: " << scale_max)
            return;
        }
        else
        {
            //printDBG("[shape_found] Shape Found And Passed")
            //printDBG("[shape_found]  * passed: " << scale)
            //printDBG("[shape_found]  * scale_min: " << scale_min << "; scale_max: " << scale_max)
        }
        // Enforce the gravity vector: convert shape into a up is up frame
        float ori = R_GRAVITY_THETA;
        rectifyAffineTransformationUpIsUp(a11, a12, a21, a22); // Helper
        std::vector<float> submaxima_oris;
        if(hesPar.rotation_invariance)
        {
            bool passed = this->localizeKeypointOrientation(
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
                submaxima_oris.push_back(R_GRAVITY_THETA + M_TAU / 24.0);
                submaxima_oris.push_back(R_GRAVITY_THETA - M_TAU / 24.0);
            }
        }

        //printDBG("[onAffShapeFound] Found " << submaxima_oris.size() << " orientations")
        global_c1++;
        global_nmulti_ori += submaxima_oris.size() - 1;
        // push a keypoint for every orientation found
        for (int i = 0; i < submaxima_oris.size(); i++)
        {
            ori = submaxima_oris[i];
            global_nkpts++;
            // sample the patch (populates this->patch)
            // (from affine.cpp)
            if(!this->normalizeAffine(this->image, x, y, s, a11, a12, a21, a22, ori)) 
            {
                this->push_new_keypoint(x, y, s, a11, a12, a21, a22, ori, type, response);
            }
        }
        //else std::cout << global_nkpts << std::endl;
        
    }
    // END void onAffineShapeFound
    //------------------------------------------------------------
    void push_new_keypoint(float x, float y, float s, float a11, float a12, float a21, float a22, float ori, int type, float response)
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
        float ori = R_GRAVITY_THETA;
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
        //normalizeAffine does the job of ptool.get_warped_patch, but uses a class variable to store the output (messy)
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
        float sigma0 = (magnitudes.rows / 2) * .4;
        float sigma1 = (magnitudes.cols / 2) * .4;
        cv::Mat gauss_weights;
        make_2d_gauss_patch_01(magnitudes.rows, magnitudes.cols, sigma0, sigma1, gauss_weights);
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
                nbins, M_TAU, 0.0);

        Histogram<float> wrapped_hist = htool::wrap_histogram(hist);
        std::vector<float> submaxima_xs, submaxima_ys;
        // inplace wrap histogram (because orientations are circular)
        htool::hist_edges_to_centers(wrapped_hist); 
        // Compute orientation as maxima of wrapped histogram
        const float maxima_thresh = this->hesPar.ori_maxima_thresh;

        htool::hist_interpolated_submaxima(
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

            define CV_8U   0
            define CV_8S   1
            define CV_16U  2
            define CV_16S  3
            define CV_32S  4
            define CV_32F  5
            define CV_64F  6

             keys = 'CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F'.split()
             print(ut.dict_str(dict(zip(keys, ut.dict_take(cv2.__dict__, keys)))))
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
//http://nbviewer.ipython.org/github/pv/SciPy-CookBook/blob/master/ipython/Ctypes.ipynb
#ifdef __cplusplus
extern "C" {
#endif

// Python binds to extern C code
#define PYHESAFF extern HESAFF_EXPORT

typedef void*(*allocer_t)(int, int*);


PYHESAFF int detect(AffineHessianDetector* detector)
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


PYHESAFF int get_cpp_version()
{
    return 1;
}


PYHESAFF int is_debug_mode()
{
    return DEBUG_ROTINVAR;
}


PYHESAFF int get_kpts_dim()
{
    return KPTS_DIM;
}

PYHESAFF int get_desc_dim()
{
    return DESC_DIM;
}


// reduce redundant function signature arguments
#define __HESAFF_PARAM_SIGNATURE_ARGS__ \
 int   numberOfScales,\
 float threshold,\
 float edgeEigenValueRatio,\
 int   border,\
 int   maxIterations,\
 float convergenceThreshold,\
 int   smmWindowSize,\
 float mrSize,\
 int spatialBins,\
 int orientationBins,\
 float maxBinValue,\
 float initialSigma,\
 int patchSize,\
 float scale_min,\
 float scale_max,\
 bool rotation_invariance,\
 bool augment_orientation,\
 float ori_maxima_thresh,\
 bool affine_invariance\

#define __HESAFF_PARAM_CALL_ARGS__ \
numberOfScales, threshold, edgeEigenValueRatio, border, maxIterations,\
convergenceThreshold, smmWindowSize, mrSize, spatialBins, orientationBins,\
maxBinValue, initialSigma, patchSize, scale_min, scale_max,\
rotation_invariance, augment_orientation, ori_maxima_thresh,\
affine_invariance

// new hessian affine detector
PYHESAFF AffineHessianDetector* new_hesaff_from_params(char* img_fpath, __HESAFF_PARAM_SIGNATURE_ARGS__)
{
    printDBG("making detector for " << img_fpath);
    printDBG(" * img_fpath = " << img_fpath);
    // Read in image and convert to uint8
    cv::Mat tmp = cv::imread(img_fpath);
    cv::Mat image(tmp.rows, tmp.cols, CV_32FC1, Scalar(0));
    float *imgout = image.ptr<float>(0);
    uint8 *imgin  = tmp.ptr<uint8>(0);
    for(size_t i = tmp.rows * tmp.cols; i > 0; i--)
    {
        *imgout = (float(imgin[0]) + imgin[1] + imgin[2]) / 3.0f;
        imgout++;
        imgin += 3;
    }

    // Define params
    SIFTDescriptorParams siftParams;
    PyramidParams pyrParams;
    AffineShapeParams affShapeParams;
    HesaffParams hesParams;

    // Copy Pyramid params
    pyrParams.numberOfScales            = numberOfScales;
    pyrParams.threshold                 = threshold;
    pyrParams.edgeEigenValueRatio       = edgeEigenValueRatio;
    pyrParams.border                    = border;
    pyrParams.initialSigma              = initialSigma;

    // Copy Affine Shape params
    affShapeParams.maxIterations        = maxIterations;
    affShapeParams.convergenceThreshold = convergenceThreshold;
    affShapeParams.smmWindowSize        = smmWindowSize;
    affShapeParams.mrSize               = mrSize;
    affShapeParams.initialSigma         = initialSigma;
    affShapeParams.patchSize            = patchSize;

    // Copy SIFT params
    siftParams.spatialBins              = spatialBins;
    siftParams.orientationBins          = orientationBins;
    siftParams.maxBinValue              = maxBinValue;
    siftParams.patchSize                = patchSize;

    // Copy my params
    hesParams.scale_min            = scale_min;
    hesParams.scale_max            = scale_max;
    hesParams.rotation_invariance  = rotation_invariance;
    hesParams.augment_orientation  = augment_orientation;
    hesParams.ori_maxima_thresh    = ori_maxima_thresh;
    hesParams.affine_invariance    = affine_invariance;
    // Create detector
    AffineHessianDetector* detector = new AffineHessianDetector(image, pyrParams, affShapeParams, siftParams, hesParams);
    detector->DBG_params();
    return detector;
}

// new hessian affine detector WRAPPER
PYHESAFF AffineHessianDetector* new_hesaff(char* img_fpath)
{
    // Pyramid Params
    int   numberOfScales = 3;
    float threshold = 16.0f / 3.0f;
    float edgeEigenValueRatio = 10.0f;
    int   border = 5;
    // Affine Params Shape
    int   maxIterations = 16;
    float convergenceThreshold = 0.05;
    int   smmWindowSize = 19;
    float mrSize = 3.0f * sqrt(3.0f);
    // SIFT params
    int spatialBins = 4;
    int orientationBins = 8;
    float maxBinValue = 0.2f;
    // Shared Pyramid + Affine
    float initialSigma = 1.6f;
    // Shared SIFT + Affine
    int patchSize = 41;
    // My params
    float scale_min = -1;
    float scale_max = -1;
    bool rotation_invariance = false;
    bool augment_orientation = false;
    float ori_maxima_thresh = .8;
    bool affine_invariance = true;

    AffineHessianDetector* detector = new_hesaff_from_params(img_fpath, __HESAFF_PARAM_CALL_ARGS__);
    return detector;
}

// extract descriptors from user specified keypoints
PYHESAFF void extractDesc(AffineHessianDetector* detector,
                          int nKpts, float* kpts, uint8* desc)
{
    printDBG("detector->extractDesc");
    detector->extractDesc(nKpts, kpts, desc);
    printDBG("extracted nKpts = " << nKpts);
}

// export current detections to numpy arrays
PYHESAFF void exportArrays(AffineHessianDetector* detector,
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
PYHESAFF void writeFeatures(AffineHessianDetector* detector,
                            char* img_fpath)
{
    detector->write_features(img_fpath);
}

void detectKeypoints(char* image_filename,
                     float** keypoints,
                     uint8** descriptors,
                     int* length,
                     __HESAFF_PARAM_SIGNATURE_ARGS__
                     )
{
    AffineHessianDetector* detector = new_hesaff_from_params(image_filename, __HESAFF_PARAM_CALL_ARGS__);
    detector->DBG_params();
    *length = detector->detect();
    *keypoints = new float[(*length)*KPTS_DIM];
    *descriptors = new uint8[(*length)*DESC_DIM];
    detector->exportArrays((*length), *keypoints, *descriptors);
    //may need to add support for "use_adaptive_scale" and "nogravity_hack" here (needs translation from Python to C++ first)
    delete detector;
}


PYHESAFF void detectKeypointsList(int num_filenames,
                                  char** image_filename_list,
                                  float** keypoints_array,
                                  uint8** descriptors_array,
                                  int* length_array,
                                  __HESAFF_PARAM_SIGNATURE_ARGS__
                                  )
{
    // Maybe use this implimentation instead to be more similar to the way
    // pyhesaff calls this library?
    int index;
    #pragma omp parallel for private(index)
    for(index = 0; index < num_filenames; ++index)
    {
        char* image_filename = image_filename_list[index];
        AffineHessianDetector* detector =
            new_hesaff_from_params(image_filename, __HESAFF_PARAM_CALL_ARGS__);
        detector->DBG_params();
        int length = detector->detect();
        length_array[index] = length;
        keypoints_array[index] = new float[length * KPTS_DIM];
        descriptors_array[index] = new uint8[length * DESC_DIM];
        exportArrays(detector, length, keypoints_array[index], descriptors_array[index]);
        delete detector;
    }
}

PYHESAFF void detectKeypointsList1(int num_filenames,
                                   char** image_filename_list,
                                   float** keypoints_array,
                                   uint8** descriptors_array,
                                   int* length_array,
                                   __HESAFF_PARAM_SIGNATURE_ARGS__
                                   )
{
    int index;
    #pragma omp parallel for private(index)
    for(index = 0; index < num_filenames; ++index)
    {
        detectKeypoints(image_filename_list[index],
                        &(keypoints_array[index]),
                        &(descriptors_array[index]),
                        &(length_array[index]),
                        __HESAFF_PARAM_CALL_ARGS__);
    }
}
#ifdef __cplusplus
}
#endif
// END PYTHON BINDINGS
//----------------------------------------------


//-------------------------------
// int main
// * program entry point for command line use if we build the executable
int main(int argc, char **argv)
{
    /*
    CommandLine:
         ./unix_build.sh --fast && ./build/hesaffexe /home/joncrall/.config/utool/star.png
         ./unix_build.sh --fast && ./build/hesaffexe /home/joncrall/.config/utool/star.png
         sh mingw_build.sh --fast
         build/hesaffexe /home/joncrall/.config/utool/star.png -rotation_invariance
    */
    const char* about_message = "\nUsage: hesaffexe image_name.png kpts_file.txt\nDescribes elliptical keypoints (with gravity vector) given in kpts_file.txt using a SIFT descriptor.\n\n";
    // Parser Reference: http://docs.opencv.org/trunk/modules/core/doc/command_line_parser.html
    //const char* keys =
    //"{help h usage ? |       | print this message   }"
    //"{@img_fpath     |       | image for detection  }"
    //"{@output_path   |.      | output file path     }"
    //"{rotation_invariance | false | rotation invariance  }"
    //"{N count        |100    | count of objects     }"
    //;
    //cv::CommandLineParser parser(argc, argv, keys);
    //parser.about(about_message);
    //if (parser.has("help"))
    //{
    //    parser.printMessage();
    //    return 0;
    //}
    //bool rotation_invariance = parser.get<bool>("rotation_invariance");
    //String img_fpath = parser.get<String>("img_fpath");

    //if (argc <= 1)
    //{
    //    parser.printParams();
    //    return 0;
    //}

    //if (!parser.check())
    //{
    //    parser.printErrors();
    //    return 0;
    //}    
    
    if(argc > 1)
    {
        printDBG("main()");
        char* img_fpath = argv[1];
        int nKpts;
        AffineHessianDetector* detector = new_hesaff(img_fpath);
        //detector->hesPar.rotation_invariance = true;
            //(argc > 2) ? atoi(argv[2]) : true;
        detector->DBG_params();
        nKpts = detect(detector);
        writeFeatures(detector, img_fpath);
        std::cout << "[main] nKpts: " << nKpts << std::endl;
        std::cout << "[main] global_nkpts: " << global_nkpts << std::endl;
        std::cout << "[main] global_c1: " << global_c1 << std::endl;
        std::cout << "[main] global_c2: " << global_c2 << std::endl;
    }
    else
    {
        printf("%s", about_message);
    }
}
