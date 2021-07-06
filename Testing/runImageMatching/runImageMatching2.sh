#!/bin/bash
execPath=".."
sourcePath="../../Data/Image_0.mhd"
targetPath="../../Data/Image_2.mhd"
numStep=10
lpower=6
truncX=16
truncY=16
truncZ=1 # if 2D, pass z as 1
sigma=0.02
alpha=3
gamma=1.0
maxIter=50
stepSizeGD=5.0e-2 # Choose smaller stepsize, e.g. 5.0e-3 for MAP 
mType=0; # 0: Host; 1: DEVICE
flag=1; #0:MAP; 1: Hieriarchical model
sampleNumber=10;
 
${execPath}/ImageMatchingTest ${sourcePath} ${targetPath} ${numStep} ${lpower} ${truncX} ${truncY} ${truncZ} ${sigma} ${alpha} ${gamma} ${maxIter} ${stepSizeGD} ${mType} ${flag} ${sampleNumber}
