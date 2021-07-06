#include "GeodesicShooting.h"
#include <time.h>

int main(int argc, char** argv)
{
  char I0Path[100];
  char I1Path[100];

  int argi = 1; 
  strcpy(I0Path, argv[argi++]);
  strcpy(I1Path, argv[argi++]);
  int numStep = atoi(argv[argi++]);
  int lpower = atoi(argv[argi++]);
  int truncX = atoi(argv[argi++]);
  int truncY = atoi(argv[argi++]);
  int truncZ = atoi(argv[argi++]);
  float sigma = atof(argv[argi++]);
  float alpha = atof(argv[argi++]);
  float gamma = atof(argv[argi++]);
  int maxIter = atoi(argv[argi++]);
  float stepSizeGD = atof(argv[argi++]);
  int memType = atoi(argv[argi++]);
  int flag = atoi(argv[argi++]);
  int sample_num = atoi(argv[argi++]);

  MemoryType mType;
  // runs on CPU or GPU
  if (memType == 0)
    mType = MEM_HOST;
  else 
    mType = MEM_DEVICE;
	
  // read data
  Image3D *I0, *I1;
  I0 = new Image3D(mType);
  I1 = new Image3D(mType);

  ITKFileIO::LoadImage(*I0, I0Path);
  ITKFileIO::LoadImage(*I1, I1Path);
  
  // access parameter
  GridInfo grid = I0->grid();
  Vec3Di mSize = grid.size();

  int fsx = mSize.x;
  int fsy = mSize.y;
  int fsz = mSize.z;

  // precalculate low frequency location
  if (truncX % 2 == 0) truncX -= 1; // set last dimension as zero if it is even
  if (truncY % 2 == 0) truncY -= 1;
  if (truncZ % 2 == 0) truncZ -= 1;
  
  FftOper *fftOper = new FftOper(alpha, gamma, lpower, grid, truncX, truncY, truncZ); 

  fftOper->FourierCoefficient();

  // forward shooting
  FieldComplex3D *v0 = new FieldComplex3D(truncX, truncY, truncZ);
  Field3D *v0Spatial = new Field3D(grid, mType);
  FieldComplex3D *gradv = new FieldComplex3D(truncX, truncY, truncZ);

  GeodesicShooting *geodesicshooting = new GeodesicShooting(fftOper, mType, numStep);
  Opers::Copy(*(geodesicshooting->I0), *I0);
  Opers::Copy(*(geodesicshooting->I1), *I1);
  geodesicshooting->sigma = sigma;
  
  clock_t t;
  t = clock();
  if (flag == 1)
      geodesicshooting->ImageMatching_HyperPrior_MCEM(*v0, *gradv, maxIter, stepSizeGD,*fftOper, flag, sample_num);
  else 
      geodesicshooting->ImageMatching_MAP(*v0, *gradv, maxIter, stepSizeGD,*fftOper, flag);
  t = clock() - t;
  printf("It took me %d clicks (%f seconds).\n",t,((float)t)/CLOCKS_PER_SEC);
	
  ITKFileIO::SaveImage(*(geodesicshooting->deformIm), "deformIm.mhd");
  fftOper->fourier2spatial(*v0Spatial, *v0);
  ITKFileIO::SaveField(*v0Spatial, "v0Spatial.mhd");

  delete fftOper;
  delete geodesicshooting;
  delete I0;
  delete I1;
  delete v0;
  delete v0Spatial;
  delete gradv;
}
