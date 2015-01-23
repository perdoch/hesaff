int global_nkpts = 0;
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

#ifdef DEBUG_HESAFF
#undef DEBUG_HESAFF
#endif
#define DEBUG_HESAFF

#ifdef DEBUG_HESAFF
#define printDBG(msg) std::cerr << "[hesaff.c] " << msg << std::endl;
#define write(msg) std::cerr << msg;
#else
#define printDBG(msg);
#endif


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

#define USE_ORI  // developing rotational invariance
#ifdef USE_ORI
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
#ifdef USE_ORI
    float ori;
#endif
    float response;
    int type;
    uint8 desc[DESC_DIM];
};


void rotate_downwards(float &a11, float &a12, float &a21, float &a22)
{
    //same as rectify_up_is_up but doest remove scale
    double a = a11, b = a12, c = a21, d = a22;
    double absdet_ = std::abs(a * d - b * c);
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
    a11 = invA_.at<float>(0, 0);
    a12 = invA_.at<float>(0, 1);
    a21 = invA_.at<float>(1, 0);
    a22 = invA_.at<float>(1, 1);
    // Rectify it (maintain scale)
    rotate_downwards(a11, a12, a21, a22);
}


cv::Mat invA_to_invE(float &a11, float &a12, float &a21, float &a22, float& s, float& desc_factor)
{
    float sc = desc_factor * s;
    cv::Mat invA = (cv::Mat_<float>(2, 2) << a11, a12, a21, a22);

    //-----------------------
    // Convert invA to invE format
    SVD svd_invA(invA, SVD::FULL_UV);
    float *diagA = (float *)svd_invA.w.data;
    diagA[0] = 1.0f / (diagA[0] * diagA[0] * sc * sc);
    diagA[1] = 1.0f / (diagA[1] * diagA[1] * sc * sc);
    cv::Mat invE = svd_invA.u * cv::Mat::diag(svd_invA.w) * svd_invA.u.t();
    return invE;
}

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
#ifdef USE_ORI
            kpts[rowk + 5] = k.ori;
#endif

#ifdef DEBUG_HESAFF
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
#ifdef USE_ORI
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
        // A circular keypoint is detected. Adpat its shape to an ellipse
        findAffineShape(blur, x, y, s, pixelDistance, type, response);
    }

    void extractDesc(int nKpts, float* kpts, uint8* desc)
    {
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
#ifdef USE_ORI
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
#ifdef DEBUG_HESAFF
            if(fx == 0)
            {
                //printDBG(" [extract_desc]    sc = "  << sc);
                //printDBG(" [extract_desc] iabcd = [" << ia << ", " << ib << ", " << ic << ", " << id << "] ");
                //printDBG(" [extract_desc]    xy = (" <<  x << ", " <<  y << ") ");
                //printDBG(" [extract_desc]    ab = [" << a11 << ", " << a12 << ",
                //printDBG(" [extract_desc]    cd =  " << a21 << ", " << a22 << "] ");
                //printDBG(" [extract_desc]     s = " << s);
            }
#endif
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
            responce - 
            iters - 

         */
        // type can be one of:

        // check if detected keypoint is within scale thresholds
        float scale_min = hesPar.scale_min;
        float scale_max = hesPar.scale_max;
        float scale = AffineShape::par.mrSize * s;
        // negative thresholds turn the threshold test off
        if((scale_min > 0 && scale < scale_min) || (scale_max > 0 && scale < scale_max))
        {
            // failed scale threshold
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
        if(hesPar.adapt_rotation)
        {
            if (!this->findKeypointsDirection(this->image, x, y, s, a11, a12, a21, a22, ori))
            {
                return;
            }
        }
        else
        {
            ori = R_GRAVITY_THETA;
        }
        global_nkpts++;
        // sample the patch (populates this->patch)
        if(!this->normalizeAffine(this->image, x, y, s, a11, a12, a21, a22, ori))  // affine.cpp
        {
            global_c1++;
            // compute SIFT and append new keypoint and descriptor
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
            #ifdef USE_ORI
            k.ori = ori;
            #endif
            this->populateDescriptor(k.desc, 0);
            //this->keys.push_back(k);
        }
        else if(hesPar.adapt_rotation)
        {
            this->keys.push_back(Keypoint());  // For debugging push back a fake keypoint
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
            k.ori = -1;
        }
        //else std::cout << global_nkpts << std::endl;
        
    }
    // END void onAffineShapeFound
    //------------------------------------------------------------
    float findKeypointsDirection(const cv::Mat& img, float x, float y,
                                 float s, 
                                 float a11, float a12,
                                 float a21, float a22,
                                 float &ori)
    {
        /*"""
        Args: 
            img : an image
            pat : a keypoints image patch

        OutVars:
            ori : dominant gradient orientation
         
        Returns: 
            bool : success flag
        """*/
        global_c2++;

        // Enforce that the shape is pointing down and sample the patch when the
        // orientation is the gravity vector 
        rectifyAffineTransformationUpIsUp(a11, a12, a21, a22);
        #ifdef DEBUG_HESAFF
        assert(ori == R_GRAVITY_THETA);
        assert(a12 == 0);
        #endif 
        // sample the patch (populates this->patch)
        if(this->normalizeAffine(img, x, y, s, a11, a12, a21, a22, ori))
        {
             // normalizerAffine is located in affine.cpp
             // normalizeAffine can fail if the keypoint is out of
             // bounds (consider adding an exception-based mechanism to
             // discard the keypoint?)
            return false;
        }
        //this->DBG_dump_patch("down_patch", this->patch);
        // Warp elliptical keypoint region in image into a (cropped) unit circle
        //normalizeAffine does the job of ptool.get_warped_patch, but uses a class variable to store the output (messy)
        // Compute gradients
        cv::Mat xgradient(this->patch.rows, this->patch.cols, this->patch.depth());
        cv::Mat ygradient(this->patch.rows, this->patch.cols, this->patch.depth());
        computeGradient(this->patch, xgradient, ygradient);
        //this->DBG_dump_patch("xgradient", xgradient);
        //this->DBG_dump_patch("ygradient", ygradient);
        // Compute orientation
        cv::Mat orientations;
        computeOrientation<float>(xgradient, ygradient, orientations);
        inplace_map(ensure_0toTau<float>, orientations.begin<float>(), orientations.end<float>());
        //cv::Mat orientations01 = orientations.mul(255.0 / 6.28);
        //this->DBG_dump_patch("orientations01", orientations01);
        // Compute magnitude
        cv::Mat magnitudes;
        computeMagnitude<float>(xgradient, ygradient, magnitudes);
        //this->DBG_dump_patch("magnitudes", magnitudes);
        // TODO: weight by a gaussian
        // Compute orientation histogram, splitting votes using linear interpolation
        const int nbins = 36;
        Histogram<float> hist = computeInterpolatedHistogram<float>(orientations.begin<float>(), orientations.end<float>(),
                                magnitudes.begin<float>(), magnitudes.end<float>(),
                                nbins);
        // wrap histogram (because orientations are circular)
        Histogram<float> wrapped_hist = htool::wrap_histogram(hist);
        htool::hist_edges_to_centers(wrapped_hist); // inplace modification
        // Compute orientation as maxima of wrapped histogram
        float submaxima_x, submaxima_y;
        htool::hist_interpolated_submaxima(wrapped_hist, submaxima_x, submaxima_y);
        float submax_ori = submaxima_x; //will change if multiple submaxima are returned
        //submax_ori -= M_GRAVITY_THETA; // adjust for 0 being downward
        submax_ori += M_GRAVITY_THETA; // adjust for 0 being downward
        submax_ori = ensure_0toTau<float>(submax_ori); //will change if multiple submaxima are returned
        printDBG("[find_ori] submax_ori = " << submax_ori)
        // populate outvar
        ori = submax_ori;
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

    void DBG_dump_patch(std::string str_name, cv::Mat& dbgpatch)
    {
        /*
        CommandLine:
             ./unix_build.sh --fast && ./build/hesaffexe /home/joncrall/.config/utool/star.png
             ./unix_build.sh --fast && ./build/hesaffexe /home/joncrall/.config/utool/star.png


        References:
            http://docs.opencv.org/modules/core/doc/basic_structures.html#mat-depth

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
        make_str(patch_fpath, "patches/" << str_name << "_" << this->keys.size() << ".png");
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
        cv::namedWindow(patch_fpath, cv::WINDOW_NORMAL);
        cv::Mat dbgpatch_;
        dbgpatch.convertTo(dbgpatch_, CV_8U);
        cv::imshow(patch_fpath, dbgpatch_);
        cv::waitKey(0);
        cv::imwrite(patch_fpath, dbgpatch);
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
        printDBG("+---");
        printDBG("|   xy = (" <<  x << ", " <<  y << ") ");
        printDBG("| invV = [(" << iv11 << ", " << iv12 << "), ");
        printDBG("|         (" << iv21 << ", " << iv22 << ")] ");
        printDBG("|  ori = " << ori);
        printDBG("L___");
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
    printDBG("nKpts = " << nKpts);
    return nKpts;
}


PYHESAFF int get_kpts_dim()
{
    return KPTS_DIM;
}

PYHESAFF int get_desc_dim()
{
    return DESC_DIM;
}

// new hessian affine detector
PYHESAFF AffineHessianDetector* new_hesaff_from_params(char* img_fpath,
        // Pyramid Params
        int   numberOfScales,
        float threshold,
        float edgeEigenValueRatio,
        int   border,
        // Affine Params Shape
        int   maxIterations,
        float convergenceThreshold,
        int   smmWindowSize,
        float mrSize,
        // SIFT params
        int spatialBins,
        int orientationBins,
        float maxBinValue,
        // Shared Pyramid + Affine
        float initialSigma,
        // Shared SIFT + Affine
        int patchSize,
        // My Params
        float scale_min,
        float scale_max,
        bool rotation_invariance)
{
    printDBG("making detector for " << img_fpath);
    printDBG("make hesaff. img_fpath = " << img_fpath);
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
    hesParams.adapt_rotation       = rotation_invariance;

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
    printDBG("affShapeParams.scale_min            = " << scale_min);
    printDBG("affShapeParams.scale_max            = " << scale_max);
    printDBG("affShapeParams.rotation_invariance  = " << rotation_invariance);
    // Create detector
    AffineHessianDetector* detector = new AffineHessianDetector(image, pyrParams, affShapeParams, siftParams, hesParams);
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

    AffineHessianDetector* detector = new_hesaff_from_params(img_fpath,
                                      numberOfScales, threshold, edgeEigenValueRatio, border,
                                      maxIterations, convergenceThreshold, smmWindowSize, mrSize,
                                      spatialBins, orientationBins, maxBinValue, initialSigma, patchSize,
                                      scale_min, scale_max, rotation_invariance);
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
    printDBG("detector->write_features");
    detector->write_features(img_fpath);
}

void detectKeypoints(char* image_filename,
                     float** keypoints,
                     uint8** descriptors,
                     int* length,
                     // Pyramid Params
                     int   numberOfScales,
                     float threshold,
                     float edgeEigenValueRatio,
                     int   border,
                     // Affine Params Shape
                     int   maxIterations,
                     float convergenceThreshold,
                     int   smmWindowSize,
                     float mrSize,
                     // SIFT params
                     int spatialBins,
                     int orientationBins,
                     float maxBinValue,
                     // Shared Pyramid + Affine
                     float initialSigma,
                     // Shared SIFT + Affine
                     int patchSize,
                     // My Params
                     float scale_min,
                     float scale_max,
                     bool rotation_invariance)
{
    AffineHessianDetector* detector = new_hesaff_from_params(image_filename,
                                      numberOfScales, threshold, edgeEigenValueRatio, border,
                                      maxIterations, convergenceThreshold, smmWindowSize, mrSize,
                                      spatialBins, orientationBins, maxBinValue, initialSigma,
                                      patchSize, scale_min, scale_max, rotation_invariance);
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
                                  // Pyramid Params
                                  int   numberOfScales,
                                  float threshold,
                                  float edgeEigenValueRatio,
                                  int   border,
                                  // Affine Params Shape
                                  int   maxIterations,
                                  float convergenceThreshold,
                                  int   smmWindowSize,
                                  float mrSize,
                                  // SIFT params
                                  int spatialBins,
                                  int orientationBins,
                                  float maxBinValue,
                                  // Shared Pyramid + Affine
                                  float initialSigma,
                                  // Shared SIFT + Affine
                                  int patchSize,
                                  // My Params
                                  float scale_min,
                                  float scale_max,
                                  bool rotation_invariance)
{
    // Maybe use this implimentation instead to be more similar to the way
    // pyhesaff calls this library?
    int index;
    #pragma omp parallel for private(index)
    for(index = 0; index < num_filenames; ++index)
    {
        char* image_filename = image_filename_list[index];
        AffineHessianDetector* detector =
            new_hesaff_from_params(image_filename, numberOfScales,
                                   threshold, edgeEigenValueRatio, border, maxIterations,
                                   convergenceThreshold, smmWindowSize, mrSize,
                                   spatialBins, orientationBins, maxBinValue, initialSigma,
                                   patchSize, scale_min, scale_max, rotation_invariance);
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
                                   // Pyramid Params
                                   int   numberOfScales,
                                   float threshold,
                                   float edgeEigenValueRatio,
                                   int   border,
                                   // Affine Params Shape
                                   int   maxIterations,
                                   float convergenceThreshold,
                                   int   smmWindowSize,
                                   float mrSize,
                                   // SIFT params
                                   int spatialBins,
                                   int orientationBins,
                                   float maxBinValue,
                                   // Shared Pyramid + Affine
                                   float initialSigma,
                                   // Shared SIFT + Affine
                                   int patchSize,
                                   // My Params
                                   float scale_min,
                                   float scale_max,
                                   bool rotation_invariance)
{
    int index;
    #pragma omp parallel for private(index)
    for(index = 0; index < num_filenames; ++index)
    {
        detectKeypoints(image_filename_list[index],
                        &(keypoints_array[index]),
                        &(descriptors_array[index]),
                        &(length_array[index]),
                        numberOfScales, threshold,
                        edgeEigenValueRatio, border, maxIterations,
                        convergenceThreshold, smmWindowSize, mrSize, spatialBins,
                        orientationBins, maxBinValue, initialSigma, patchSize,
                        scale_min, scale_max, rotation_invariance);
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
    if(argc > 1)
    {
        printDBG("main()");
        char* img_fpath = argv[1];
        int nKpts;
        AffineHessianDetector* detector = new_hesaff(img_fpath);
        detector->hesPar.adapt_rotation = (argc > 2) ? atoi(argv[2]) : true;
        nKpts = detect(detector);
        writeFeatures(detector, img_fpath);
        std::cout << "nKpts: " << nKpts << std::endl;
        std::cout << "global_nkpts: " << global_nkpts << std::endl;
        std::cout << "global_c1: " << global_c1 << std::endl;
        std::cout << "global_c2: " << global_c2 << std::endl;
    }
    else
    {
        printf("\nUsage: ell_desc image_name.png kpts_file.txt\nDescribes elliptical keypoints (with gravity vector) given in kpts_file.txt using a SIFT descriptor.\n\n");
    }
}
