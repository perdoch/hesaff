#ifndef _HESAFF_DLLDEFINES_H
#define _HESAFF_DLLDEFINES_H

#ifdef WIN32
#ifndef snprintf
#define snprintf _snprintf
#endif // ndef sprintf
#endif // WIN32

#define HESAFF_EXPORT // ????
#ifndef FOO_DLL  // ???
#ifdef HESAFF_EXPORTS // EXPORTS??? ... No need on mingw
#define HESAFF_EXPORT __declspec(dllexport)
#else
//#define HESAFF_EXPORT __declspec(dllimport)
#endif
#else
#define HESAFF_EXPORT
#endif // FOO_DLL // ???

// TODO : use either adapt_rotation or rotation_invariance, but not both

struct HesaffParams
{
    float scale_min;     // minimum scale threshold
    float scale_max;     // maximum scale threshold
    float ori_maxima_thresh;  // threshold for orientation invaraince
    bool rotation_invariance;  // are we assuming the gravity vector?
    bool adapt_rotation;
    bool adapt_scale;
    bool affine_invariance;
    bool augment_orientation;

    HesaffParams()
    {
        scale_min = -1;
        scale_max = -1;
        ori_maxima_thresh = .8; 
        rotation_invariance = false; //remove in favor of adapt_rotation?
        augment_orientation = false; //remove in favor of adapt_rotation?
        adapt_rotation = false;
        adapt_scale = false;
        affine_invariance = true;  // if false uses circular keypoints 
    }
};

#endif //_HESAFF_DLLDEFINES_H
