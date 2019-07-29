#ifndef __ORIENTATION_H__
#define __ORIENTATION_H__

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <vector>
#include <numeric>
#include <iostream>
#include "helpers.h"

//#include <cv.h>
#include <opencv2/opencv.hpp>
using namespace cv;

#ifndef M_TAU
#define M_TAU 6.28318f
#endif

// DUPLICATE
#define DEBUG_ROTINVAR 0

#define make_str(str_name, stream_input) \
    std::string str_name;\
{std::stringstream tmp_sstm;\
    tmp_sstm << stream_input;\
    str_name = tmp_sstm.str();\
};

#if DEBUG_ROTINVAR
#define printDBG_ORI(msg) std::cerr << "[ori.c] " << msg << std::endl;
#define write(msg) std::cerr << msg;
#else
#define printDBG_ORI(msg);
#endif


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
    std::vector<T> centers;
    T step;
    T range_min;
    T range_max;
};


template <class T> void print_vector(std::vector<T> vec, const char* name="vec")
{
    //std::cout << name << " = [";
    std::cout << "[ori.c] >>> " << name << " = [";
    for(int i = 0; i < vec.size(); ++i)
    {
        std::cout <<
            //std::setprecision(8) <<
            //std::setprecision(2) <<
            vec[i] << ", " ;
        if (i % 6 == 0 && i > 0)
        {
            std::cout << std::endl << "...     ";
        }
    }
    std::cout << "]" << std::endl;
}

template <class T> void show_hist_submaxima(const Histogram<T>& hist, float maxima_thresh=.8)
{
    /*
     python -m pyhesaff._pyhesaff --test-test_rot_invar --show
    */
    // Uses python command via system call to debug visually
    make_str(basecmd, "python -m vtool.histogram --test-show_hist_submaxima --show");
    printDBG_ORI("SHOWING HIST SUBMAXIMA WITH PYTHON");
    make_str(cmdstr, basecmd <<
            " --hist=\"" << hist.data << "\"" <<
            " --edges=\"" << hist.edges << "\"" <<
            " --maxima_thesh " << maxima_thresh <<
            " --title cpporihist" <<
            //" --legend"
            ""
            );
    //printDBG_ORI(cmdstr.c_str());
    run_system_command(cmdstr);
}


template <class T> T ensure_0toTau(T x)
{
    T _tau = static_cast<T>(M_TAU);
    if(x < 0)return ensure_0toTau(x+_tau);
    else if(x >= _tau)return ensure_0toTau(x-_tau);
    else return x;
}

template <class UnaryFn, class Iterator> void inplace_map(UnaryFn fn, Iterator begin, Iterator end)
{
    for(Iterator iter = begin; iter != end; ++iter)
    {
        *iter = fn(*iter);
    }
}

// requires C++11, manually instantiated in wrap histogram since unsure if this much of C++11 is supported by the compilers this project is intended to work with
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

template <class T> T python_modulus(T numer, T denom)
{
    /*
    modulus like it works in python
    */
    T result = static_cast<T>(fmod(static_cast<double>(numer), static_cast<double>(denom)));
    if (result < 0)
    {
        result = denom + result;
    }
    return result;
}

template <class T, class Iterator> Histogram<T> computeInterpolatedHistogram(
        Iterator data_beg, Iterator data_end,
        Iterator weight_beg, Iterator weight_end,
        int nbins, T range_max, T range_min)
{
    /*
    FIXME: We assume this histogram is wrapped

    CommandLine:
        python -m pyhesaff test_rot_invar --show
        ./unix_build.sh --fast && ./build/hesaffexe ~/.config/utool/star.png
        python -m vtool.patch --test-find_dominant_kp_orientations

     */
    // Input: Iterators over orientations and magnitudes and the number of bins to discretize the orientation domain
    const bool interpolation_wrap = true;
    const T start = range_min;
    const T stop = range_max;
    const T step = (stop - start) / T(nbins + static_cast<T>(interpolation_wrap));
    const T half_step = step / static_cast<T>(2.0);
    const T data_offset = start + half_step;
    // debug info
    printDBG_ORI("nbins = " << nbins)
    printDBG_ORI("step = " << step)
    printDBG_ORI("half_step = " << half_step)
    printDBG_ORI("data_offset = " << data_offset)
    // data offset should be 0, but is around for more general case of
    // interpolated histogram
    //assert(data_offset == 0);
    //
    Histogram<T> hist;  // structure of bins and edges
    hist.step = step;
    hist.range_min = range_min;
    hist.range_max = range_max;
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
        // Read the item and determine which bins it should vote into
        T data = *data_iter;
        T weight = *weight_iter;

        T fracBinIndex = (data - data_offset) / step;
        int left_index = int(floor(fracBinIndex));
        int right_index = left_index + 1;
        T right_alpha = fracBinIndex - T(left_index);  // get interpolation weight between 0 and 1
        // Wrap around indices
        //    static int nprint = 0;
        //    bool doprint = false && (left_index < 0 || right_index >= nbins);
        //    if (doprint)
        //    {
        //        printDBG_ORI("+----")
        //        printDBG_ORI("range = " << left_index << ", " << right_index)
        //    }
        left_index  = python_modulus(left_index, nbins);
        right_index = python_modulus(right_index, nbins);
        //    if (doprint)
        //    {
        //        printDBG_ORI("range = " << left_index << ", " << right_index)
        //        printDBG_ORI("L___")
        //    }
        //    nprint++;
        // Linear Interpolation of gradient magnitude votes over orientation
        // bins (maybe do quadratic)
        hist.data[left_index]  += weight * (1 - right_alpha);
        hist.data[right_index] += weight * (right_alpha);

    }
    return hist;
}

// from htool (vtool.histogram)
namespace htool
{

template <class T> Histogram<T> wrap_histogram(const Histogram<T>& input)
{
    // FIXME; THIS NEEDS INFORMATION ABOUT THE DISTANCE FROM THE LAST BIN
    // TO THE FIRST. IT IS OK AS LONG AS ALL STEPS ARE EQUAL, BUT IT IS NOT
    // GENERAL
    std::vector<T> tmp;
    tmp.resize(input.edges.size());
    std::adjacent_difference(input.edges.begin(), input.edges.end(), tmp.begin());
    int low = tmp[0], high = tmp[tmp.size() - 1];
    Histogram<T> output;
    output.step = input.step;
    output.range_min = input.range_min;
    output.range_max = input.range_max;
    //vector_concat(output.data, input.data[input.data.size()-1], input.data, input.data[0]);
    output.data.push_back(input.data[input.data.size() - 1]);
    for(typename std::vector<T>::const_iterator iter = input.data.begin(); iter != input.data.end(); ++iter) {
        output.data.push_back(*iter);
    }
    output.data.push_back(input.data[0]);
    //vector_concat(output.edges, input.edges[0]-low, input.edges, input.edges[input.edges.size()-1]+high);
    output.edges.push_back(input.edges[0] - input.step);
    for(typename std::vector<T>::const_iterator iter = input.edges.begin(); iter != input.edges.end(); ++iter) {
        output.edges.push_back(*iter);
    }
    output.edges.push_back(input.edges[input.edges.size() - 1] + input.step);

    //    print_vector<T>(output.data, "wrapped_hist");
    //    print_vector<T>(output.edges, "wrapped_edges");
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
    return (a + b) / 2;
}
}

template <class T> void hist_edges_to_centers(Histogram<T>& hist)
{
    pairwise_accumulate<T, T(T, T)>(hist.centers, hist.edges, hist_edges_to_centers_lambdas::average);
}

template <class T> void hist_argmaxima(Histogram<T> hist, std::vector<int>& argmaxima_list, float maxima_thresh=0.8f)
{
    int size = static_cast<int>(hist.data.size()) - 2; // the edge points aren't to be counted, due to being only there to make interpolation simpler
    // The first and last bins are duplicates so we dont need to look at those
    int argmax = 1;
    T maxval = hist.data[1]; // initialize
    for (int i = 1; i < hist.data.size() - 1; i++)
    {
        if (hist.data[i] > maxval)
        {
            maxval = hist.data[i];
            argmax = i;
        }
    }
    float thresh_val = maxval * maxima_thresh;
    // Finds all maxima within maxima_thresh of the maximum
    for (int i = 1; i < hist.data.size() - 1; i++)
    {
        T prev = hist.data[i - 1];
        T curr = hist.data[i];
        T next = hist.data[i + 1];
        // Test if a maxima and above a threshold
        if (curr > prev && curr > next && curr > thresh_val)
        {
            argmaxima_list.push_back(i);
            //T maxima_x = (hist.edges[i] + hist.edges[i + 1]);
            //maxima_xs.push_back(maxima_x);
            //maxima_ys.push_back(hist.data[i]);
        }
    }
    //float* data = &(hist.data[1]);
    //// Pack our histogram structure into an opencv histogram
    //cv::Mat cvhist(1, &size, cv::DataType<T>::type, data);
    //cv::Point argmax2d;
    //// Find index of the maxima ONLY
    //cv::minMaxLoc(cvhist, NULL, NULL, NULL, &argmax2d);
    //argmaxima = argmax2d.y + 1;
    //maxima_x = hist.edges[argmaxima];
    //maxima_y = hist.data[argmaxima];
    //printDBG_ORI("maxima_xs, maxima_ys " << maxima_xs << ", " << maxima_ys)
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

template <class T> void vector_take(
        const std::vector<T>& item_list,
        const std::vector<int>& index_list,
        std::vector<T>& item_sublist)
{
    //item_sublist.resize(index_list.size());
    for (int i = 0; i < index_list.size(); i++)
    {
        int index = index_list[i];
        T item = item_list[index];
        item_sublist.push_back(item);
    }
}


template <class T> void argsubmaxima(const Histogram<T>& hist, std::vector<T>& submaxima_xs, std::vector<T>& submaxima_ys, float maxima_thresh=.8)
{
    std::vector<int> argmaxima_list;
    // TODO: Currently this returns only one maxima, maybe later incorporate
    // multiple maxima.
    // Get the discretized bin maxima
    hist_argmaxima(hist, argmaxima_list, maxima_thresh);
    //    printDBG_ORI("Argmaxima:");
    //    std::vector<T> maxima_ys;
    //    print_vector(maxima_ys, "maxima_ys");
    //    vector_take(hist.data, argmaxima_list, maxima_ys);
    //    //vector_take(hist.edges, argmaxima_list, maxima_xs)
    //    print_vector(argmaxima_list, "argmaxima_list");
    //    print_vector(maxima_ys, "maxima_ys");
    for (int i = 0; i < argmaxima_list.size(); i++)
    {
        int argmaxima = argmaxima_list[i];
        T submaxima_x, submaxima_y;
        // Interpolate the maxima using 2nd order Taylor polynomials
        interpolate_submaxima(argmaxima, hist, submaxima_x, submaxima_y);
        submaxima_xs.push_back(submaxima_x);
        submaxima_ys.push_back(submaxima_y);
    }
    //    print_vector(submaxima_xs, "submaxima_xs");
    //    print_vector(submaxima_ys, "submaxima_ys");
}

}

//from ptool (vtool.patch)
//void getWarpedPatch(const cv::Mat& chip, Keypoint kpt); //AffineShape::normalizeAffine does this



#endif // __ORIENTATION_H__
