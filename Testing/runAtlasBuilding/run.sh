#!/bin/sh
MPIDIR="OPENMPI_PATH"
numProcessor=5

execPath="BINARY_PATH"
dataPath="DATA_PATH"
resultPath="RESULT_PATH"
suffix=".mhd"
numStep=10
lpower=3
truncX=32
truncY=32
truncZ=32 # if 2D, pass z as 1
sigma=0.03
alpha=3.0
gamma=1.0
maxIter=1
stepSizeGD=5.0e-2
mType=0 # 0: Host; 1: DEVICE
EmIter=100

#================================================
# collect all the filename in the directory
#AtlasFileDir=$dataPath/"*"$suffix
AtlasFileDir=$(find $dataPath -name "DATA_NAME")
AtlasResultDir=$resultPath/"v0"
if [ ! -d $AtlasResultDir ]
then
    mkdir $AtlasResultDir
fi

echo "Atlas building..."

${MPIDIR}/bin/mpirun -np ${numProcessor} ${execPath}/AtlasBuildingTest ${suffix} ${AtlasResultDir} ${numStep} ${lpower} ${truncX} ${truncY} ${truncZ} ${sigma} ${alpha} ${gamma} ${maxIter} ${stepSizeGD} ${mType} ${EmIter} ${AtlasFileDir}
