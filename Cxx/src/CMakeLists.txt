SET(Base_SRC
    Base/FftOper.cxx
    Base/FieldComplex3D.cxx
    Base/Armalib.cxx
    Base/MPIlib.cxx
    )

ADD_LIBRARY(GeodesicShooting ${Base_SRC} Alg/GeodesicShooting.cxx)
TARGET_LINK_LIBRARIES(GeodesicShooting ${PyCA_LIB} ${FFTWF_LIB} ${FFTWF_THREADS_LIB})

ADD_LIBRARY(Tpca ${Base_SRC} Alg/Tpca.cxx)
TARGET_LINK_LIBRARIES(Tpca ${PyCA_LIB} ${ARMADILLO_LIBRARIES} ${FFTWF_LIB} ${FFTWF_THREADS_LIB})
