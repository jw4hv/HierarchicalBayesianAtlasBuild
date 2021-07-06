ADD_EXECUTABLE(AtlasBuildingTest AtlasBuildingTest.cxx)
TARGET_LINK_LIBRARIES(AtlasBuildingTest ${PyCA_LIB} GeodesicShooting ${FFTWF_LIB} ${FFTWF_THREADS_LIB})

ADD_EXECUTABLE(ImageMatchingTest ImageMatchingTest.cxx)
TARGET_LINK_LIBRARIES(ImageMatchingTest ${PyCA_LIB} GeodesicShooting ${FFTWF_LIB} ${FFTWF_THREADS_LIB})

