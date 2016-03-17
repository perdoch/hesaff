

//PYHESAFF void detectKeypointsList(int num_fpaths,
//                                  char** image_fpath_list,
//                                  float** keypoints_array,
//                                  uint8** descriptors_array,
//                                  int* length_array,
//                                  __HESAFF_PARAM_SIGNATURE_ARGS__
//                                  )
//{
//    assert(0);  // do not use
//    // Maybe use this implimentation instead to be more similar to the way
//    // pyhesaff calls this library?
//    int index;
//    #pragma omp parallel for private(index)
//    for(index = 0; index < num_fpaths; ++index)
//    {
//        char* image_filename = image_fpath_list[index];
//        AffineHessianDetector* detector =
//            new_hesaff_fpath(image_filename, __HESAFF_PARAM_CALL_ARGS__);
//        detector->DBG_params();
//        int length = detector->detect();
//        length_array[index] = length;
//        // TODO: shouldn't python be doing this allocation?
//        keypoints_array[index] = new float[length * KPTS_DIM];
//        descriptors_array[index] = new uint8[length * DESC_DIM];
//        exportArrays(detector, length, keypoints_array[index], descriptors_array[index]);
//        delete detector;
//    }
//}

//void detectKeypoints(char* image_filename,
//                     float** keypoints,
//                     uint8** descriptors,
//                     int* length,
//                     __HESAFF_PARAM_SIGNATURE_ARGS__
//                     )
//{
//    AffineHessianDetector* detector = new_hesaff_fpath(image_filename, __HESAFF_PARAM_CALL_ARGS__);
//    detector->DBG_params();
//    *length = detector->detect();
//    // TODO: shouldn't python be doing this allocation?
//    *keypoints = new float[(*length)*KPTS_DIM];
//    *descriptors = new uint8[(*length)*DESC_DIM];
//    detector->exportArrays((*length), *keypoints, *descriptors);
//    //may need to add support for "use_adaptive_scale" and "nogravity_hack" here (needs translation from Python to C++ first)
//    delete detector;
//}

//typedef void*(*allocer_t)(int, int*);


//const PYHESAFF char* cmake_build_type()
//{
//    // References:
//    // http://stackoverflow.com/questions/14883853/ctypes-return-a-string-from-c-function
//    char *build_type = (char*) malloc(sizeof(char) * (10 + 1));
//    #ifdef CMAKE_BUILD_TYPE
//    //char hello[] = CMAKE_BUILD_TYPE
//    strcpy(build_type, "testb1");
//    #else
//    strcpy(build_type, "testb2");
//    #endif
//    return build_type;
//}

//PYHESAFF char* free_char(char* malloced_char)
//{
//    // need to free anything malloced here
//    free(malloced_char);
//}
