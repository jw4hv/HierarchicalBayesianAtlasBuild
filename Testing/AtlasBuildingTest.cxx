#include "GeodesicShooting.h"
#include "MPIlib.h"
#include "Vec2D.h"

int main(int argc, char** argv)
{
  char integMethod[100];
  char suffix[100];
  char resultDir[100];
  char filename[100];

  int argi = 1;
  strcpy(suffix, argv[argi++]);
  strcpy(resultDir, argv[argi++]);
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
  int EmIter = atoi(argv[argi++]); 
  
  MemoryType mType;
  // runs on CPU or GPU
  if (memType == 0)
    mType = MEM_HOST;
  else 
    mType = MEM_DEVICE;

  // Initialize MPI.
  int p, id, stride, tSubj, iSubj; //total # of subjects, # of subjects on each process
  MPI::Init(argc, argv);

  // Get the number of processes.
  p = MPI::COMM_WORLD.Get_size();
  // Determine the rank of this process.
  id = MPI::COMM_WORLD.Get_rank();
  // Get the number of subjects on each process
  tSubj= argc-argi;
  stride = tSubj / p;
  if (id == p-1)
    iSubj = stride + tSubj % p;
  else
    iSubj = stride;

  // read data
  Image3D *I1;
  I1 = new Image3D(mType);
  ITKFileIO::LoadImage(*I1, argv[argi]); // read the first image to get information

  // access parameter
  GridInfo grid = I1->grid();
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
    
  // parameters for atlas building 
  // create two arrays for MPI
  int nImVox = fsx*fsy*fsz;
  Field3D *v0Spatial = new Field3D(grid, mType);
  FieldComplex3D *v0 = new FieldComplex3D(truncX, truncY, truncZ);
  FieldComplex3D *gradv = new FieldComplex3D(truncX, truncY, truncZ);
  FieldComplex3D** storeV0 = new FieldComplex3D* [iSubj];
  Image3D** storeI1 = new Image3D* [iSubj];
  Image3D* sumSplatI = new Image3D(grid, mType);
  Image3D* sumSplatOnes = new Image3D(grid, mType);
  double sumIEnergy;

  GeodesicShooting *geodesicshooting = new GeodesicShooting(fftOper, mType, numStep);

  cout << "initialize atlas on process " << id << endl;

  Opers::SetMem(*(geodesicshooting->I0), 0.0);
  Vec2D<float> maxmin;

  for (int i = 0; i < iSubj; i++)
    {
      storeV0[i] = new FieldComplex3D(truncX, truncY, truncZ);
      storeI1[i] = new Image3D(grid, mType);
      ITKFileIO::LoadImage(*storeI1[i], argv[argi+id*stride+i]);

      // normalize image intensity to [0, 1]
      Opers::MaxMin(maxmin, *storeI1[i]);
      Opers::MulC_I(*storeI1[i], 1.0/maxmin[0]);
      Opers::MaxMin(maxmin, *storeI1[i]);

      cout << "max is: " << maxmin[0] << "  min is:  " << maxmin[1] << endl;  

      Opers::Add_I(*(geodesicshooting->I0), *storeI1[i]);
    }
  // initialization for template and v0
  MPI::COMM_WORLD.Allreduce(MPI::IN_PLACE, (geodesicshooting->I0)->get(),
  			    nImVox, MPI::DOUBLE, MPI::SUM);
  Opers::MulC_I(*(geodesicshooting->I0), 1.0/static_cast<float>(tSubj));
  geodesicshooting->sigma = sigma;

  // atlas estimation
  for (int j = 0; j < EmIter; j++)
    {
      Opers::SetMem(*sumSplatI, 0.0);
      Opers::SetMem(*sumSplatOnes, 0.0);
      sumIEnergy = 0.0;

      for (int i = 0; i < iSubj; i++)
  	{
  	  Opers::Copy(*(geodesicshooting->I1), *storeI1[i]);
  	  geodesicshooting->ImageMatching(*storeV0[i], *gradv, maxIter, stepSizeGD);
  	  Opers::Add_I(*sumSplatI, *(geodesicshooting->splatI));
  	  Opers::Add_I(*sumSplatOnes, *(geodesicshooting->splatOnes));
  	  sumIEnergy += geodesicshooting->IEnergy;
  	}
      // update atlas
      MPI::COMM_WORLD.Allreduce(MPI::IN_PLACE, sumSplatI->get(),
  				nImVox, MPI::FLOAT, MPI::SUM);
      MPI::COMM_WORLD.Allreduce(MPI::IN_PLACE, sumSplatOnes->get(),
  				nImVox, MPI::FLOAT, MPI::SUM);
      MPI::COMM_WORLD.Allreduce(MPI::IN_PLACE, &sumIEnergy,
  				1, MPI::DOUBLE, MPI::SUM);
      Opers::Div(*(geodesicshooting->I0), *sumSplatI, *sumSplatOnes);

      if (id == 0)
  	cout << "IEnergy is," << sumIEnergy << endl;
    }

  // write out results
  if (id == 0)
    {
      sprintf(filename, "%s/template%s", resultDir, suffix);
      ITKFileIO::SaveImage(*(geodesicshooting->I0), filename);
    }

  // save initial velocity v0
  for (int i = 0; i < iSubj; i++)
    {
      fftOper->fourier2spatial(*v0Spatial, *storeV0[i]);
      sprintf(filename, "%s/v0_%d%s", resultDir, id*stride+i, suffix);
      ITKFileIO::SaveField(*v0Spatial, filename);
      delete storeV0[i];
      delete storeI1[i];
    }
    
  delete [] storeV0;
  delete [] storeI1;
  delete I1;
  delete v0Spatial;
  delete v0;
  delete fftOper;
  delete geodesicshooting;
  delete sumSplatI;
  delete sumSplatOnes;
  
  MPI::Finalize(); // terminate MPI
}
