#ifndef _HESAFF_DLLDEFINES_H
#define _HESAFF_DLLDEFINES_H

#ifdef WIN32
    #ifndef snprintf
    #define snprintf _snprintf
    #endif
#endif

#define HESAFF_EXPORT
#ifndef FOO_DLL
    // No need on mingw
    #ifdef HESAFF_EXPORTS
        #define HESAFF_EXPORT __declspec(dllexport)
    #else
        //#define HESAFF_EXPORT __declspec(dllimport)
    #endif
#else
#define HESAFF_EXPORT
#endif



#endif //_HESAFF_DLLDEFINES_H
