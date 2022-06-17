#ifndef _HESAFF_DLLDEFINES_H
#define _HESAFF_DLLDEFINES_H

#ifdef WIN32
#ifndef snprintf_s
#define snprintf_s _snprintf_s
#endif // snprintf_s
#endif // WIN32

// Define HESAFF_EXPORTED for any platform
// References: https://atomheartother.github.io/c++/2018/07/12/CPPDynLib.html
#if defined _WIN32 || defined __CYGWIN__
  #ifdef HESAFF_WIN_EXPORT
    // Exporting...
    #ifdef __GNUC__
      #define HESAFF_EXPORTED __attribute__ ((dllexport))
    #else
      #define HESAFF_EXPORTED __declspec(dllexport) // Note: actually gcc seems to also supports this syntax.
    #endif
  #else
    #ifdef __GNUC__
      #define HESAFF_EXPORTED __attribute__ ((dllimport))
    #else
      #define HESAFF_EXPORTED __declspec(dllimport) // Note: actually gcc seems to also supports this syntax.
    #endif
  #endif
  #define HESAFF_NOT_EXPORTED
#else
  #if __GNUC__ >= 4
    #define HESAFF_EXPORTED __attribute__ ((visibility ("default")))
    #define HESAFF_NOT_EXPORTED  __attribute__ ((visibility ("hidden")))
  #else
    #define HESAFF_EXPORTED
    #define HESAFF_NOT_EXPORTED
  #endif
#endif


// TODO : use either adapt_rotation or rotation_invariance, but not both

struct HesaffParams
{
    float scale_min;     // minimum scale threshold
    float scale_max;     // maximum scale threshold
    float ori_maxima_thresh;  // threshold for orientation invariance
    bool rotation_invariance;  // are we assuming the gravity vector?
    bool adapt_rotation;
    bool adapt_scale;
    bool affine_invariance;
    bool augment_orientation;
    bool only_count;

    HesaffParams()
    {
        scale_min = -1.0f;
        scale_max = -1.0f;
        ori_maxima_thresh = 0.8f;
        rotation_invariance = false; //remove in favor of adapt_rotation?
        augment_orientation = false; //remove in favor of adapt_rotation?
        adapt_rotation = false;
        adapt_scale = false;
        affine_invariance = true;  // if false uses circular keypoints
        only_count = false;
    }
};

#endif //_HESAFF_DLLDEFINES_H
