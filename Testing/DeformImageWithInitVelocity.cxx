#include "GeodesicShooting.h"
#include <time.h>

struct params
{
  int numStep;
  int lpower;
  int truncX;
  int truncY;
  int truncZ;
  float alpha;
  float gamma;
  int memType;
  MemoryType mType;
  bool doRK4;
};

int main(int argc, char** argv)
{
  char I0Path[500];
  char v0Path[500];

  int argi = 1;
  strcpy(I0Path, argv[argi++]);
  strcpy(v0Path, argv[argi++]);
  
  params parms;
  parms.numStep = atoi(argv[argi++]);
  parms.lpower = atoi(argv[argi++]);
  parms.truncX = atoi(argv[argi++]);
  parms.truncY = atoi(argv[argi++]);
  parms.truncZ = atoi(argv[argi++]);
  parms.alpha = atof(argv[argi++]);
  parms.gamma = atof(argv[argi++]);
  parms.memType = atoi(argv[argi++]);
  parms.doRK4 = atoi(argv[argi++]) > 0;

  // precalculate low frequency location
  if (parms.truncX % 2 == 0) parms.truncX -= 1; // set last dimension as zero if it is even
  if (parms.truncY % 2 == 0) parms.truncY -= 1;
  if (parms.truncZ % 2 == 0) parms.truncZ -= 1;

  // runs on CPU or GPU
  if (parms.memType == 0)
    parms.mType = MEM_HOST;
  else 
    parms.mType = MEM_DEVICE;
	  
  // read data
  Image3D *I0;
  I0 = new Image3D(parms.mType);

  ITKFileIO::LoadImage(*I0, I0Path);

  Field3D *v0Spatial = new Field3D(parms.mType);
  ITKFileIO::LoadField(*v0Spatial, v0Path);
  GridInfo grid = v0Spatial->grid();
  Vec3Di mSize = grid.size();

  int fsx = mSize.x;
  int fsy = mSize.y;
  int fsz = mSize.z;

  FieldComplex3D *v0 = new FieldComplex3D(parms.truncX, parms.truncY, parms.truncZ);
  FieldComplex3D *grad = new FieldComplex3D(parms.truncX, parms.truncY, parms.truncZ);
  Field3D *phiinv = new Field3D(grid, parms.mType);

  clock_t t;
  t = clock();

  FftOper *fftOper = new FftOper(parms.alpha, parms.gamma, parms.lpower, grid,
                                 parms.truncX, parms.truncY, parms.truncZ);
  fftOper->FourierCoefficient();
  fftOper->spatial2fourier(*v0, *v0Spatial);

  GeodesicShooting *geodesicshooting = new GeodesicShooting(fftOper, parms.mType, parms.numStep, parms.doRK4);

  Opers::Copy(*(geodesicshooting->I0), *I0);

  geodesicshooting->fwdIntegration(*v0, *grad);

  Opers::HtoV(*phiinv, *(geodesicshooting->phiinv));

  t = clock() - t;
  printf("It took me %lu clicks (%f seconds).\n",t,((float)t)/CLOCKS_PER_SEC);

  ITKFileIO::SaveImage(*(geodesicshooting->deformIm), "deformIm.mhd");
  ITKFileIO::SaveField(*phiinv, "phiInv.mhd");

  delete fftOper;
  delete geodesicshooting;
  delete I0;
  delete v0;
  delete v0Spatial;
  delete grad;
  delete phiinv;

}
