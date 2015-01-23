/*
 * Copyright (C) 2008-12 Michal Perdoch
 * All rights reserved.
 *
 * This file is part of the HessianAffine detector and is made available under
 * the terms of the BSD license (see the COPYING file).
 *
 */

#ifndef __HELPERS_H__
#define __HELPERS_H__

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv.h>
#include <cmath>
#include <vector>
#include <numeric>

#ifndef M_TAU
#define M_TAU 6.28318
#endif

// DUPLICATE
#ifdef DEBUG_HELPERS
#undef DEBUG_HELPERS
#endif
#define DEBUG_HELPERS

#ifdef DEBUG_HELPERS
#define printDBG2(msg) std::cerr << "[helpers.c] " << msg << std::endl;
#define write(msg) std::cerr << msg;
#else
#define printDBG2(msg);
#endif

void solveLinear3x3(float *A, float *b);
bool getEigenvalues(float a, float b, float c, float d, float &l1, float &l2);
void invSqrt(float &a, float &b, float &c, float &l1, float &l2);
void computeGaussMask(cv::Mat &mask);
void computeCircularGaussMask(cv::Mat &mask);
void rectifyAffineTransformationUpIsUp(float *U);
void rectifyAffineTransformationUpIsUp(float &a11, float &a12, float &a21, float &a22);
void rotateAffineTransformation(float &a11, float &a12, float &a21, float &a22, float &theta);
bool interpolate(const cv::Mat &im, float ofsx, float ofsy, float a11, float a12, float a21, float a22, cv::Mat &res);
bool interpolateCheckBorders(const cv::Mat &im, float ofsx, float ofsy, float a11, float a12, float a21, float a22, const cv::Mat &res);
void photometricallyNormalize(cv::Mat &image, const cv::Mat &weight_mask, float &sum, float &var);

cv::Mat gaussianBlur(const cv::Mat input, float sigma);
void gaussianBlurInplace(cv::Mat &inplace, float sigma);
cv::Mat doubleImage(const cv::Mat &input);
cv::Mat halfImage(const cv::Mat &input);

double getTime();

bool almost_eq(float, float);
void computeGradient(const cv::Mat &img, cv::Mat &gradx, cv::Mat &grady);


// from ltool (vtool.linalg)
template<class T> cv::Mat get3x3Translation(T x, T y)
{
    cv::Mat t = (cv::Mat_<T>(3, 3) <<
                 1, 0, x,
                 0, 1, y,
                 0, 0, 1);
    return t;
}
template<class T> cv::Mat get3x3Rotation(T theta)
{
    T c = cos(theta);
    T s = sin(theta);
    cv::Mat m = (cv::Mat_<T>(3, 3) <<
                 c, -s, 0,
                 s,  c, 0,
                 0,  0, 1);
    return m;
}
template<class T> cv::Mat get3x3Scale(T sx, T sy)
{
    cv::Mat s = (cv::Mat_<T>(3, 3) <<
                 sx,  0, 0,
                 0, sy, 0,
                 0,  0, 1);
    return s;
}

template <class T, class BinaryFn> void matrix_map_2to1(BinaryFn fn, cv::InputArray in1, cv::InputArray in2, cv::OutputArray out)
{
    cv::Mat in1_m = in1.getMat();
    cv::Mat in2_m = in2.getMat();
    const int width = in1_m.cols;
    const int height = in1_m.rows;
    out.create(in1_m.size(), in1_m.type());
    cv::Mat out_m = out.getMat();
    for(int r = 0; r < height; ++r)
        for(int c = 0; c < width; ++c)
        {
            out_m.at<T>(r, c) = fn(in1_m.at<T>(r, c), in2_m.at<T>(r, c));
        }
}

namespace computeOrientationMagnitude_lambdas
{
template <class T> T xatan2(T x, T y)
{
    return atan2(y, x);
}
template <class T> T distance(T x, T y)
{
    return sqrt((x*x)+(y*y));
}
}

template <class T> void computeOrientation(cv::InputArray xgradient, cv::InputArray ygradient, cv::OutputArray orientations)
{
    /*cv::Mat xg = xgradient.getMat();
    cv::Mat yg = ygradient.getMat();
    const int width = xg.cols;
    const int height = xg.rows;
    orientations.create(xg.size(), xg.type());
    cv::Mat ori = orientations.getMat();
    for(int r = 0; r < height; ++r)
        for(int c = 0; c < width; ++c)
            {
                ori.at<T>(r, c) = atan2(yg.at<T>(r, c), xg.at<T>(r, c));
            }*/
    matrix_map_2to1<T, T(T, T)>(computeOrientationMagnitude_lambdas::xatan2<T>, xgradient, ygradient, orientations);
}


template <class T> void computeMagnitude(cv::InputArray xgradient, cv::InputArray ygradient, cv::OutputArray magnitudes)
{
    matrix_map_2to1<T, T(T, T)>(computeOrientationMagnitude_lambdas::distance<T>, xgradient, ygradient, magnitudes);
}

template <class T, class Iterator> T minimumElement(Iterator begin, Iterator end)
{
    T rv = *begin;
    for(Iterator iter = begin; iter != end; ++iter)
    {
        if(*iter < rv)rv = *iter;
    }
    return rv;
}

template <class T, class Iterator> T maximumElement(Iterator begin, Iterator end)
{
    T rv = *begin;
    for(Iterator iter = begin; iter != end; ++iter)
    {
        if(*iter > rv)rv = *iter;
    }
    return rv;
}

// helper function for computeInterpolatedHistogram
/*template <class T> int computeBin(T elem, T min, T max, T step)
{
    // uses linear search, possibly upgrade to binary search later?
    //  WRONG: split vote based on ratio of distance
    int rv = 0;
    for(T x = min; x < max; x += step)
    {
        if(elem < x)break;
        rv++;
    }
    return rv;
}*/

template <class T> struct Histogram
{
    std::vector<T> data;
    std::vector<T> edges;
};

template <class T> T ensure_0toTau(T x)
{
    if(x < 0)return ensure_0toTau(x+M_TAU);
    else if(x >= M_TAU)return ensure_0toTau(x-M_TAU);
    else return x;
}

template <class UnaryFn, class Iterator> void inplace_map(UnaryFn fn, Iterator begin, Iterator end)
{
    for(Iterator iter = begin; iter != end; ++iter)
    {
        *iter = fn(*iter);
    }
}

template <class T, class Iterator> Histogram<T> computeInterpolatedHistogram(
        Iterator data_beg, Iterator data_end, 
        Iterator weight_beg, Iterator weight_end,
        int nbins)
{
    /*

    CommandLine:
        ./unix_build.sh --fast && ./build/hesaffexe /home/joncrall/.config/utool/star.png
        python -m vtool.patch --test-find_dominant_kp_orientations
       

     */
    // Input: Iterators over orientaitons and magnitudes and the number of bins to discretize the orientation domain
    const T start = 0;
    const T stop = M_TAU;
    const T step = (stop - start) / T(nbins);
    const T half_step = step / 2.0;
    const T data_offset = start + half_step;
    // debug info
    printDBG2("nbins = " << nbins)
    printDBG2("step = " << step)
    printDBG2("half_step = " << half_step)
    printDBG2("data_offset = " << data_offset)
    //float ans = fmod(-.5, M_TAU);
    //printDBG2("-.5 MOD tau = " << ans);
    //int num_list[] = {-8, -1, 0, 1, 2, 6, 7, 29};
    //for (int i=0; i < 8; i++)
    //{
    //    int num = num_list[i];
    //    float res_fmod = fmod(float(num), M_TAU);
    //    float pymodop_res = fmod(float(num), M_TAU);
    //    if (pymodop_res < 0)
    //    {
    //        pymodop_res = M_TAU + pymodop_res;
    //    }

    //    printf("num=%4d, res_py=%5.2f, res_fmod=%5.2f\n", num, pymodop_res, res_fmod);
    //}

    // data offset should be 0, but is around for more general case of
    // interpolated histogram
    assert(data_offset == 0);
    // 
    Histogram<T> hist;  // structure of bins and edges
    hist.data.resize(nbins);  // Allocate space for bins
    for(int i = 0; i <= nbins; ++i)
    {
        hist.edges.push_back((T(i) * step) + start);
    }
    // For each pixel orientation and magnidue
    for(
        Iterator data_iter = data_beg, weight_iter = weight_beg; 
        (data_iter != data_end) && (weight_iter != weight_end);
        ++data_iter, ++weight_iter
       )
    {
        T data = *data_iter;
        T weight = *weight_iter;
        T fracBinIndex = (data - data_offset) / step;
        int left_index = int(floor(fracBinIndex));
        int right_index = left_index + 1;
        T right_alpha = fracBinIndex - left_index;  // get interpolation weight between 0 and 1
        // Wrap around indicies
        // TODO: optimize
        left_index  = int(htool::python_modulus(float(left_index), float(nbins)));
        right_index = int(htool::python_modulus(float(right_index), float(nbins)));
        printDBG2("left_index = " << left_index)
        printDBG2("right_index = " << right_index)
        printDBG2("-----")
        // Linear Interpolation of gradient magnitude votes over orientation bins (maybe do quadratic)
        hist.data[left_index]     += weight * (1 - right_alpha);
        hist.data[right_index] += weight * (right_alpha);
    }
    return hist;
}

// requires C++11, manually instantiated in wrap_histogram since unsure if this much of C++11 is supported by the compilers this project is intended to work with
/*
template <class T> void vector_concat(std::vector<T>& out) {}
template <class T, class... Rest> void vector_concat(std::vector<T>& out, T elem, Rest... rest) { out.push_back(elem); vector_concat(out, rest...); }
template <class T, class ContainerT, class... Rest> void vector_concat(std::vector<T>& out, const ContainerT& elems, Rest... rest)
{
    for(typename ContainerT::const_iterator iter = elems.begin(); iter != elems.end(); ++iter)
    {
        out.push_back(*iter);
    }
    vector_concat(out, rest...);
}
*/

template <class T, class BinaryFunction> void pairwise_accumulate(std::vector<T>& out, const std::vector<T>& in, BinaryFunction fn)
{
    typename std::vector<T>::const_iterator iter = in.begin();
    T prev = *iter;
    ++iter;
    for(; iter != in.end(); ++iter)
    {
        out.push_back(fn(prev, *iter));
        prev = *iter;
    }
}

template <class T> std::ostream& operator << (std::ostream& os, const std::vector<T>& c)
{
    os << "[";
    for(typename std::vector<T>::const_iterator iter = c.begin(); iter != c.end(); ++iter)
    {
        os << *iter;
        if((iter+1) != c.end()) os << ", ";
    }
    os << "]";
    return os;
}

template <class T> std::ostream& operator << (std::ostream& os, const Histogram<T>& h)
{
    os << "Histogram(" << h.data << ", " << h.edges << ")";
    return os;
}

// from htool (vtool.histogram)
namespace htool
{

float python_modulus(float numer, float denom)
{
    /* module like it works in python */
    float result = fmod(numer, denom);
    if (result < 0)
    {
        result = denom + result;
    }
    return result;
}
template <class T> Histogram<T> wrap_histogram(const Histogram<T>& input)
{
    std::vector<int> tmp;
    tmp.resize(input.edges.size());
    std::adjacent_difference(input.edges.begin(), input.edges.end(), tmp.begin());
    int low = tmp[0], high = tmp[tmp.size()-1];
    Histogram<T> output;
    //vector_concat(output.data, input.data[input.data.size()-1], input.data, input.data[0]);
    output.data.push_back(input.data[input.data.size()-1]);
    for(typename std::vector<T>::const_iterator iter = input.data.begin(); iter != input.data.end(); ++iter) {
        output.data.push_back(*iter);
    }
    output.data.push_back(input.data[0]);
    //vector_concat(output.edges, input.edges[0]-low, input.edges, input.edges[input.edges.size()-1]+high);
    output.edges.push_back(input.edges[0]-low);
    for(typename std::vector<T>::const_iterator iter = input.edges.begin(); iter != input.edges.end(); ++iter) {
        output.edges.push_back(*iter);
    }
    output.edges.push_back(input.edges[input.edges.size()-1]+high);
    return output;
}

//template <class T> std::vector<T> linspace_with_endpoint(T start, T stop, int num):
//{
//    [> simulate np.linspace
//    if num == 1:
//        return array([start], dtype=dtype)
//    step = (stop-start)/float((num-1))
//    y = _nx.arange(0, num, dtype=dtype) * step + start
//    y[-1] = stop
//    */ 
//    std::vector<T> domain;
//    if (num == 1)
//    {
//        domain.push_back(start);
//    }
//    float step = (stop - start) / float((num - 1.0))
//    for (int i=0; i < (num - 1); i++)
//    {
//        domain.push_back(i * step + start);
//    }
//    domain.push_back(stop)
//    return domain
//}

//void makeCvHistFromHistogram(Histogram<float>& hist, CvHistogram& cvHist);

namespace hist_edges_to_centers_lambdas
{
template <class T> T average(T a, T b)
{
    return (a+b)/2;
}
}

template <class T> void hist_edges_to_centers(Histogram<T>& hist)
{
    std::vector<T> centers;
    pairwise_accumulate<T, T(T, T)>(centers, hist.edges, hist_edges_to_centers_lambdas::average);
    hist.edges = centers;
}

template <class T> void hist_argmaxima(Histogram<T> hist, T& maxima_x, T& maxima_y, int& argmaxima)
{
    /*CvHistogram cvHist;
    makeCvHistFromHistogram(hist, cvHist);
    cvGetMinMaxHistValue(&cvHist, NULL, NULL, NULL, &argmaxima);
    maxima_x = hist.edges[argmaxima];
    maxima_y = hist.data[argmaxima];*/
    int size = hist.data.size() - 2; // the edge points aren't to be counted, due to being only there to make interpolation simpler
    float* data = &(hist.data[1]);
    // Pack our histogram structure into an opencv histogram
    cv::Mat cvhist(1, &size, cv::DataType<T>::type, data);
    cv::Point argmax2d;
    // Find index of the maxima ONLY
    cv::minMaxLoc(cvhist, NULL, NULL, NULL, &argmax2d);
    argmaxima = argmax2d.y + 1;
    maxima_x = hist.edges[argmaxima];
    maxima_y = hist.data[argmaxima];;
}

template <class T> void maxima_neighbors(int argmaxima, const Histogram<T>& hist, std::vector<cv::Point_<T> >& points)
{
    for(int i = -1; i <= 1; i++)
    {
        int j = argmaxima+i;
        T y = hist.data[j];
        T x = hist.edges[j];
        points.push_back(cv::Point_<T>(x, y));
    }
}

template <class T> void interpolate_submaxima(int argmaxima, const Histogram<T>& hist, T& submaxima_x, T& submaxima_y)
{
    std::vector<cv::Point_<T> > points;
    maxima_neighbors(argmaxima, hist, points);
    T x1, y1, x2, y2, x3, y3;
    x1 = points[0].x;
    y1 = points[0].y;
    x2 = points[1].x;
    y2 = points[1].y;
    x3 = points[2].x;
    y3 = points[2].y;
    T denom = (x1 - x2) * (x1 - x3) * (x2 - x3);
    T A     = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom;
    T B     = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / denom;
    T C     = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom;
    T xv = -B / (2 * A);
    T yv = C - B * B / (4 * A);
    submaxima_x = xv;
    submaxima_y = yv;
}
template <class T> void hist_interpolated_submaxima(const Histogram<T>& hist, T& submaxima_x, T& submaxima_y)
{
    T maxima_x, maxima_y;
    int argmaxima;
    // TODO: Currently this returns only one maxima, maybe later incorporate multiple maxima
    // Get the discretized bin maxima
    hist_argmaxima(hist, maxima_x, maxima_y, argmaxima);
    // Interpolate the maxima using 2nd order Taylor polynomials
    interpolate_submaxima(argmaxima, hist, submaxima_x, submaxima_y);
}
}

//from ptool (vtool.patch)
//void getWarpedPatch(const cv::Mat& chip, Keypoint kpt); //AffineShape::normalizeAffine does this

#endif // __HELPERS_H__
