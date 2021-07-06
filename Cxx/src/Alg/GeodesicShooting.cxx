#include "GeodesicShooting.h"
#include <time.h>
#include <boost/version.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/polygamma.hpp>

GeodesicShooting::GeodesicShooting(FftOper* _fftOper,
 const MemoryType _mType,
 const int _numTimeSteps,
 bool _doRK4)
{
   fftOper = _fftOper;
   xDim = fftOper->truncX;
   yDim = fftOper->truncY;
   zDim = fftOper->truncZ;
   mType = _mType;
   grid = fftOper->grid;
   numTimeSteps = _numTimeSteps;
   dt = 1.0 / numTimeSteps;
   doRK4 = _doRK4;

   storeV = new FieldComplex3D* [numTimeSteps+1];
   for (int i = 0; i <= numTimeSteps; i++)
     storeV[i] = new FieldComplex3D(xDim, yDim, zDim);

 adScratch1 = new FieldComplex3D(xDim, yDim, zDim);
 adScratch2 = new FieldComplex3D(xDim, yDim, zDim);
 scratch1 = new FieldComplex3D(xDim, yDim, zDim);
 scratch2 = new FieldComplex3D(xDim, yDim, zDim);
 scratch3 = new FieldComplex3D(xDim, yDim, zDim);
 JacX = new FieldComplex3D(xDim, yDim, zDim);
 JacY = new FieldComplex3D(xDim, yDim, zDim);
 JacZ = new FieldComplex3D(xDim, yDim, zDim);

 imMatchGradient = new FieldComplex3D(xDim, yDim, zDim);
 fwd_gradvfft = new FieldComplex3D(xDim, yDim, zDim);

 scratchV1 = new Field3D(grid, mType);
 scratchV2 = new Field3D(grid,  mType);
 phiinv = new Field3D(grid, mType);

 I0 = new Image3D(grid, mType);
 I1 = new Image3D(grid, mType);
 deformIm = new Image3D(grid, mType);
 splatI = new Image3D(grid, mType);
 splatOnes = new Image3D(grid, mType);
 residualIm = new Image3D(grid, mType);

     // id matrix in Fourier domain can be preset
 identity = new Field3D(grid, mType);
 Opers::SetToIdentity(*identity);
 idxf = new float[2 * fftOper->fsxFFT * fftOper->fsy * fftOper->fsz];
 idyf = new float[2 * fftOper->fsxFFT * fftOper->fsy * fftOper->fsz];
 idzf = new float[2 * fftOper->fsxFFT * fftOper->fsy * fftOper->fsz];
 fftOper->spatial2fourier_F(idxf, idyf, idzf, *identity); 
}

GeodesicShooting::~GeodesicShooting()
{
   delete I0; I0 = NULL;
   delete I1; I1 = NULL;
   delete residualIm; residualIm = NULL;
   delete deformIm; deformIm = NULL;
   delete splatI; splatI = NULL;
   delete splatOnes; splatOnes = NULL;
   delete scratch1; scratch1 = NULL;
   delete scratch2; scratch2 = NULL;
   delete scratch3; scratch3 = NULL;
   delete adScratch1; adScratch1 = NULL;
   delete adScratch2; adScratch2 = NULL;
   delete scratchV1; scratchV1 = NULL;
   delete scratchV2; scratchV2 = NULL;
   delete phiinv; phiinv = NULL;
   delete imMatchGradient; imMatchGradient = NULL;
   delete fwd_gradvfft; fwd_gradvfft = NULL;
   delete JacX; JacX = NULL;
   delete JacY; JacY = NULL;
   delete JacZ; JacZ = NULL;
   delete identity; identity = NULL;
   delete idxf; idxf = NULL;
   delete idyf; idyf = NULL;
   delete idzf; idzf = NULL;

   for (int i = 0; i <= numTimeSteps; i++)
       {delete storeV[i]; storeV[i] = NULL;}
   delete [] storeV; storeV = NULL;
}

// ad operator
// CD: central difference
// ConvolveComplexFFT(CD*v, w)-ConvolveComplexFFT(CD*w, v))
void GeodesicShooting::ad(FieldComplex3D& advw,
   const FieldComplex3D& v,
   const FieldComplex3D& w)
{
   Jacobian(*JacX, *JacY, *JacZ, *(fftOper->CDcoeff), v); 
   fftOper->ConvolveComplexFFT(advw, 0, *JacX, *JacY, *JacZ, w);

   Jacobian(*JacX, *JacY, *JacZ, *(fftOper->CDcoeff), w); 
   fftOper->ConvolveComplexFFT(*adScratch1, 0, *JacX, *JacY, *JacZ, v);

   AddI_FieldComplex(advw, *adScratch1, -1.0);
}

// adTranspose
// spatial domain: K(Dv^T Lw + div(L*w x v))
// K*(CorrComplexFFT(CD*v^T, L*w) + TensorCorr(L*w, v) * D)
void GeodesicShooting::adTranspose(FieldComplex3D& adTransvw,
 const FieldComplex3D& v,
 const FieldComplex3D& w)
{
   Mul_FieldComplex(*adScratch1, *(fftOper->Lcoeff), w);

   JacobianT(*JacX, *JacY, *JacZ, *(fftOper->CDcoeff), v); 
   fftOper->ConvolveComplexFFT(adTransvw, 1, *JacX, *JacY, *JacZ, *adScratch1);

   fftOper->CorrComplexFFT(*adScratch2, v, *adScratch1, *(fftOper->CDcoeff));
   AddI_FieldComplex(adTransvw, *adScratch2, 1.0);

   MulI_FieldComplex(adTransvw, *(fftOper->Kcoeff));
} 


void GeodesicShooting::fwdIntegrateV(const FieldComplex3D& v0)
{
 Copy_FieldComplex(*storeV[0], v0);

 if (doRK4)
 {
   for (int i = 1; i <= numTimeSteps; i++)
   {
       // v1 = v0 - (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
       // k1
     adTranspose(*scratch1, *storeV[i-1], *storeV[i-1]);
       // partially update v1 = v0 - (dt/6)*k1
     Copy_FieldComplex(*storeV[i], *storeV[i-1]);
     AddI_FieldComplex(*storeV[i], *scratch1, -dt / 6.0);

       // k2
     Copy_FieldComplex(*scratch3, *storeV[i-1]);
     AddI_FieldComplex(*scratch3, *scratch1, -0.5*dt);
     adTranspose(*scratch2, *scratch3, *scratch3);
       // partially update v1 = v1 - (dt/3)*k2
     AddI_FieldComplex(*storeV[i], *scratch2, -dt / 3.0);

       // k3 (stored in scratch1)
     Copy_FieldComplex(*scratch3, *storeV[i-1]);
     AddI_FieldComplex(*scratch3, *scratch2, -0.5*dt);
     adTranspose(*scratch1, *scratch3, *scratch3);
       // partially update v1 = v1 - (dt/3)*k3
     AddI_FieldComplex(*storeV[i], *scratch1, -dt / 3.0);

       // k4 (stored in scratch2)
     Copy_FieldComplex(*scratch3, *storeV[i-1]);
     AddI_FieldComplex(*scratch3, *scratch1, -dt);
     adTranspose(*scratch2, *scratch3, *scratch3);
       // finish updating v1 = v1 - (dt/6)*k4
     AddI_FieldComplex(*storeV[i], *scratch2, -dt / 6.0);
 }
}
else
{
   for (int i = 1; i <= numTimeSteps; i++)
   {
	    // v0 = v0 - dt * adTranspose(v0, v0)
      Copy_FieldComplex(*storeV[i], *storeV[i-1]);
      adTranspose(*scratch1, *storeV[i], *storeV[i]);
      AddI_FieldComplex(*storeV[i], *scratch1, -dt);
  }
}

}
// forward integration
// Euler integration
// calculate phiinv by integrating backward
void GeodesicShooting::fwdIntegration(FieldComplex3D& v0,
  FieldComplex3D& fwd_gradvfft)
{
   fwdIntegrateV(v0);
     Copy_FieldComplex(v0, *storeV[numTimeSteps]); // done because prev version was updating v0

     // Opers::SetToIdentity(*phiinv);

     // for (int i = 0; i < numTimeSteps; i++) // generate phi^{-1} under left invariant metric
     // {
     // 	  fftOper->fourier2spatial(*scratchV1, *storeV[i]);
     // 	  Opers::JacobianXY(*scratchV2, *phiinv, *scratchV1); // scratchV2: D(phiinv) v_t
     // 	  Opers::Add_MulC_I(*phiinv, *scratchV2, -dt);
     // }

     scratch1->initVal(complex<float>(0.0, 0.0)); // displacement field
     for (int i = 0; i < numTimeSteps; i++) // generate phi^{-1} under left invariant metric
     {
         Jacobian(*JacX, *JacY, *JacZ, *(fftOper->CDcoeff), *scratch1); 
         fftOper->ConvolveComplexFFT(*scratch2, 0, *JacX, *JacY, *JacZ, *storeV[i]);
         AddIMul_FieldComplex(*scratch1, *scratch2, *storeV[i], -dt);
     }

     fftOper->fourier2spatial_addH(*phiinv, *scratch1, idxf, idyf, idzf);

     // calculate gradient
     Opers::ApplyH(*deformIm, *I0, *phiinv, BACKGROUND_STRATEGY_WRAP);
     Opers::Gradient(*scratchV1, *deformIm, DIFF_CENTRAL, BC_WRAP);
     Opers::Sub(*residualIm, *deformIm, *I1);
     Opers::MulMulC_I(*scratchV1, *residualIm, -1.0);
     fftOper->spatial2fourier(fwd_gradvfft, *scratchV1);

     MulI_FieldComplex(fwd_gradvfft, *(fftOper->Kcoeff));
 }

// update adjoint variable
// for backward integration
 void GeodesicShooting::bwdUpdate(FieldComplex3D& dvadj,
   const FieldComplex3D& vadj,
   const FieldComplex3D& ad, 
   const FieldComplex3D& adTrans)
 {
   for (int i = 0; i < xDim * yDim * zDim * 3; i++)
     dvadj.data[i] += dt * (vadj.data[i] - ad.data[i] + adTrans.data[i]);
}

// backward integration, reduced adjoint jacobi fields
// Euler integration
void GeodesicShooting::bwdIntegration(FieldComplex3D& dvadj,
				      FieldComplex3D& vadj) // forward gradient
{
     dvadj.initVal(complex<float>(0.0, 0.0)); // backward to t=0
     for (int i = numTimeSteps; i > 0; i--) // reduced adjoint jacobi fields
     {
       // dvadj = dvadj - dt * (-vadj + ad(v, dvadj) - adTranspose(dvadj, v));
       // dvadj = dvadj + dt * (vadj - ad(v, dvadj) + adTranspose(dvadj, v));
       // vadj = vadj + dt * adTranspose(v, vadj);

         ad(*scratch1, *storeV[i], dvadj);
         adTranspose(*scratch2, dvadj, *storeV[i]);
         bwdUpdate(dvadj, vadj, *scratch1, *scratch2);

         adTranspose(*scratch1, *storeV[i], vadj);
         AddI_FieldComplex(vadj, *scratch1, dt);
     }
 }

// Gamma function
 double GammaFunction (double k )
 {  double factorization = 1 ; 
    for (int i = k-1; i >0; i--){
        factorization *= i;
    }
    return factorization;
}

// gradient term of parameter \alpha 
double GeodesicShooting::GradientAlpha(double grad_alpha, double alpha, FieldComplex3D& newV0, 
    double const_k, double const_theta, int flag)
{
    FieldComplex3D *tempV = new FieldComplex3D(xDim, yDim, zDim);
    FieldComplex3D *lapneg = new FieldComplex3D(xDim, yDim, zDim);
    float deterterm, priorterm, deterenergy;
    double temp1, temp2 = 0; 
    const_theta = alpha/const_k;
    deterterm = 0;
    double result = 0 ; 
    deterenergy = 0;
    
    for (int i = 0; i < xDim * yDim * zDim*3; i++){
        temp1 = (fftOper->discretelap)->data[i].real();
        temp2 = (fftOper->discretelap)->data[i].real() * alpha +fftOper->gamma;
        deterterm += temp1/temp2;
        deterenergy += log(temp2);
    }
    //AlphaEnergy = -0.5*deterenergy -  ((const_k -1)*log(alpha) + alpha/const_theta + const_k* log(const_theta) + log(GammaFunction(const_k))) ;
    
    Mul_FieldComplex(*tempV, *(fftOper->discretelap), *(fftOper->Lcoeffminus));
    Mul_FieldComplex(*tempV, *tempV, newV0);
    priorterm = xDim*yDim*zDim*3*Dotprod(*tempV, newV0).real();
    deterterm = deterterm;

    if (flag == 0)
        grad_alpha =  fftOper->lpow*0.5*(priorterm - deterterm) ;
    else 
        grad_alpha =  (fftOper->lpow*0.5*(priorterm - deterterm) - ((const_k-1)/alpha + 1/const_theta));
    return grad_alpha;
}

//Apppximation of Inverse Digama function 
double GeodesicShooting::InvDigamma (double input){
    double M, Y;
    int approx_num = 3;
    if (input >= -2.22)
        M = 1.0;  
    else 
        M =0.0;
    Y = M*(exp(input) + 0.5) + (1-M)* -1./(input- boost::math::digamma(1));
    for (int i =0; i <= approx_num; i++)
        Y = Y - (boost::math::digamma(Y)- input)/boost::math::polygamma(1,Y);
    return Y;

}

// gradient term for image matching
void GeodesicShooting::Gradient(FieldComplex3D& v0,
    FieldComplex3D& gradv)
{
   Copy_FieldComplex(gradv, v0);

     // calculate Venergy
   Mul_FieldComplex(*scratch1, *(fftOper->Lcoeff), v0);
   fftOper->fourier2spatial(*scratchV1, *scratch1);
   fftOper->fourier2spatial(*scratchV2, v0);
   Opers::Dot(VEnergy, *scratchV1, *scratchV2);
   VEnergy *= 0.5;

     // forward integration
   fwdIntegration(v0, *fwd_gradvfft);

     // backward integration
     // imMatchGradient should be initialized as zero before passing
   bwdIntegration(*imMatchGradient, *fwd_gradvfft);

     // calculate Ienergy
   Opers::Sum2(IEnergy, *residualIm);
   TotalEnergy = VEnergy + 0.5/(sigma*sigma)*IEnergy;

     // gradient term
     // gradv = v0 + \tilde{v};
   AddI_FieldComplex(gradv, *imMatchGradient, 1.0/(sigma*sigma));
}

// gradient descent strategy for image matching
// adaptive stepsize
void GeodesicShooting::ImageMatching(FieldComplex3D& v0,
   FieldComplex3D& gradv,
   int maxIter,
   float stepSizeGD)
{
   float epsilon = 1.0e-10;
   float prevEnergy = 1e20;

   FieldComplex3D *currV0 = new FieldComplex3D(xDim, yDim, zDim);
   FieldComplex3D *prevV0 = new FieldComplex3D(xDim, yDim, zDim);
   FieldComplex3D *newV0 = new FieldComplex3D(xDim, yDim, zDim);
   Copy_FieldComplex(*prevV0, v0);

   for (int i = 0; i < maxIter; i++)
   {
     Copy_FieldComplex(*currV0, v0);

     Gradient(v0, gradv);

     Add_FieldComplex(*newV0, *currV0, gradv, -stepSizeGD);

     if (TotalEnergy > prevEnergy)
     {
        stepSizeGD *= 0.8;
        Copy_FieldComplex(v0, *prevV0);
    }
    else
    {
        Copy_FieldComplex(*prevV0, *currV0);
        Copy_FieldComplex(v0, *newV0);
        prevEnergy = TotalEnergy;
    }

    if (gradv.Norm() < epsilon)
        break;
}

Opers::SetMem(*residualIm, 1.0);
Opers::Splat(*splatI, *phiinv, *I1, BACKGROUND_STRATEGY_WRAP);
Opers::Splat(*splatOnes, *phiinv, *residualIm, BACKGROUND_STRATEGY_WRAP);

delete currV0; currV0 = NULL;
delete prevV0; prevV0 = NULL;
delete newV0; newV0 = NULL;
}
// Maximum a posterior approach for image matching
void GeodesicShooting::ImageMatching_MAP(FieldComplex3D& v0,
    FieldComplex3D& gradv,
    int maxIter,
    float stepSizeGD, FftOper& fftOper, int flag)
{
    float epsilon = 1.0e-10;
    float prevEnergy = 1e20;
    float maxTotal = 0;
    double stepsizealpha = 8e-4;
    pair<double, double> EnergyQue[maxIter];
    FieldComplex3D *currV0 = new FieldComplex3D(xDim, yDim, zDim);
    FieldComplex3D *prevV0 = new FieldComplex3D(xDim, yDim, zDim);
    FieldComplex3D *newV0 = new FieldComplex3D(xDim, yDim, zDim);
    FieldComplex3D *tempV = new FieldComplex3D(xDim, yDim, zDim);
    FieldComplex3D *Vsquare = new FieldComplex3D(xDim, yDim, zDim);
    FieldComplex3D *lapneg = new FieldComplex3D(xDim, yDim, zDim);

    Field3D *vtempSpatial = new Field3D(grid, mType);
    char filename[100];

    double gradalpha= 0; 
    double newalpha = 5; 
    double init_k = 9;
    double inittheta = .5;
    double Max_alpha = 40;

    Copy_FieldComplex(*prevV0, v0);
    
    for (int itr= 0; itr < maxIter; itr++)
    {
        gradalpha = GradientAlpha (gradalpha, newalpha, v0, init_k, inittheta, flag);

        //Adaptive step size 
        if (gradalpha >= Max_alpha)
            stepsizealpha *= 0.8;

        //Gradient descent for updating alpha
        newalpha = newalpha - (gradalpha)*stepsizealpha;        
        fftOper.alpha = newalpha;
        fftOper.FourierCoefficient();
        cout << "The estimation for parameter alpha is: " << newalpha <<endl;

        //Run MAP for the hierarchical posterior when flag equals to 1
        if (flag == 1)
            {inittheta = newalpha/init_k ; 
                init_k = InvDigamma (log(newalpha)-log(inittheta));}

        //Gradient descent for updating velocity, newV0 = v0 - stepSizeGD*gradv;
                Copy_FieldComplex(*currV0, v0);
                Gradient(v0, gradv);
                Add_FieldComplex(*newV0, *currV0, gradv, -stepSizeGD);

                if (TotalEnergy > prevEnergy)
                {
                    stepSizeGD *= 0.8;
                    Copy_FieldComplex(v0, *prevV0);
                }
                else
                {
                    Copy_FieldComplex(*prevV0, *currV0);
                    Copy_FieldComplex(v0, *newV0);
                    prevEnergy = TotalEnergy;
                }
                cout<<"TotalEnergy=    "<<TotalEnergy<<"  VEnergy  "<<VEnergy<<"  IEnergy  "<<IEnergy<<endl;
                if (gradv.Norm() < epsilon)
                    break;

            }

            Opers::SetMem(*residualIm, 1.0);
            Opers::Splat(*splatI, *phiinv, *I1, BACKGROUND_STRATEGY_WRAP);
            Opers::Splat(*splatOnes, *phiinv, *residualIm, BACKGROUND_STRATEGY_WRAP);

            delete currV0; currV0 = NULL;
            delete prevV0; prevV0 = NULL;
            delete newV0; newV0 = NULL;
        }

// Hierarchical Bayesian model for image matching and atlas building 
void GeodesicShooting::ImageMatching_HyperPrior_MCEM(FieldComplex3D& v0,
    FieldComplex3D& gradv,
    int maxIter,
    float stepSizeGD, FftOper& fftOper, int flag, int sample_num)
        {
            int LeapfrogSteps = 10;
            int k = 0;
            float epsilon = 1.0e-10;
            float prevEnergy = 1e20;
            float maxTotal = 0;
            double HmcStepSize = 5e-3;
            double current_K, current_U, proposed_K, proposed_U;
            pair<double, double> EnergyQue[maxIter];
            FieldComplex3D *currV0 = new FieldComplex3D(xDim, yDim, zDim);
            FieldComplex3D *prevV0 = new FieldComplex3D(xDim, yDim, zDim);
            FieldComplex3D *newV0 = new FieldComplex3D(xDim, yDim, zDim);
            FieldComplex3D *tempV = new FieldComplex3D(xDim, yDim, zDim);
            FieldComplex3D *lapneg = new FieldComplex3D(xDim, yDim, zDim);
            FieldComplex3D *grad_expect = new FieldComplex3D(xDim, yDim, zDim);
            FieldComplex3D *grad_total = new FieldComplex3D(xDim, yDim, zDim);

            Field3D *vtempSpatial = new Field3D(grid, mType);
            char filename[100];
            int rand_surplus = 1000;
            double rand_scale = 10;
            double gradalpha = 0; 
            double average_alpha = 0;

            //Initialize alpha, theta, and k paramters;
            double initalpha = 5;
            double scalar_k = 9.0 ;
            double inittheta = 0.5;

            // Saving theta and k paramters after integrating out latent variable alpha;
            double newalpha = 0 ; 
            double newtheta = 0;
            double current_q, q, current_p, p = 0;
            double alphasamples [sample_num];

            Copy_FieldComplex(*prevV0, v0);
            for (int em_itr = 0; em_itr < maxIter; em_itr++)
            {   
                average_alpha = 0;
                for (int itr= 0; itr < sample_num; itr++)
                {
                    q = initalpha; 
                    current_q = q; 
            //Initilize knetic energy with a Gaussian distribution
                    p = ((rand()%rand_surplus)/rand_scale);
                    current_p = p; 
                    current_K = 0.5*p*p; 
                    p = p - 0.5 *HmcStepSize*GradientAlpha (gradalpha, initalpha, v0, scalar_k, inittheta, flag);
                    current_U = TotalEnergy;

            // full step for p and q
            // q = q + HmcStepSize*gradp
                    for (int i = 0; i < LeapfrogSteps-1; i++)
                    {
                        q = q+p*HmcStepSize; 
                        p = p - 1.0 *HmcStepSize*GradientAlpha (gradalpha, q, v0, scalar_k, inittheta, flag);
                    }
                    q = q + p *HmcStepSize;
            // final half step for p
                    p = p - 0.5 *HmcStepSize*GradientAlpha (gradalpha, q, v0, scalar_k, inittheta, flag);
                    
                    proposed_U = TotalEnergy;
                    proposed_K = 0.5 * p * p;
                    alphasamples[itr] = q; 
                    
            //Check potential and knectic energy
            // std::cout << "current_U, " << current_U << "," <<endl;
            // std::cout << "current_K, " << current_K << ","<<endl;
            // std::cout << "proposed_U, " << proposed_U << ","<<endl;
            // std::cout << "proposed_K, " << proposed_K << ","<<endl;
                    cout<<exp((current_U-proposed_U+current_K-proposed_K)/1000) <<endl;
                    if (log((float)rand()) / RAND_MAX < exp((current_U-proposed_U+current_K-proposed_K)/1000))
                    {
                       alphasamples[itr] = q; 
                   }
                   else 
                   {
                    alphasamples[itr] = current_q; 
                }       
            // Check accepted alpha samples
            cout<<"sample candidates:"<< alphasamples[itr] <<endl; 
            }

        /*Check the averaged value of alpha samples*/
        double average_alpha = 0;
        for (k = 0; k< sample_num; k ++){
            average_alpha+=alphasamples[k];
        }
        average_alpha = average_alpha/sample_num;
        cout<<"average_alpha:" << average_alpha<< " "<<endl;

        /*Mstep for theta*/
            newtheta = average_alpha/scalar_k ; 
            inittheta = newtheta;

        /*Mstep for k*/
            scalar_k = InvDigamma (log(average_alpha) - log(inittheta));
            cout<<"k: " << scalar_k << "theta: "<<inittheta<<endl;

        /*Mstep for v_0*/
            Copy_FieldComplex(*currV0, v0);
            //Calculating gradient of v_0 from all alpha samples
            grad_total->initVal(complex<float>(0.0, 0.0));
            for (int num =0; num < sample_num; num++){
                grad_expect->initVal(complex<float>(0.0, 0.0));
                Copy_FieldComplex(*tempV, v0);
                fftOper.alpha = alphasamples[num];
                fftOper.FourierCoefficient();
                Gradient(*tempV, *grad_expect);
                AddI_FieldComplex(*grad_total, *grad_expect, 1.0);
            }
            MulC_FieldComplex(gradv, *grad_total, 1.0/sample_num);
            //Update v_0
            Add_FieldComplex(*newV0, *currV0, gradv, -stepSizeGD);


            if (TotalEnergy > prevEnergy)
            {  
                stepSizeGD *= 0.8;
                Copy_FieldComplex(v0, *prevV0);
            }
            else
            {
                Copy_FieldComplex(*prevV0, *currV0);
                Copy_FieldComplex(v0, *newV0);
                prevEnergy = TotalEnergy;
            }
        // Check the convergence of Total Energy 
        cout<<"TotalEnergy=    "<<TotalEnergy  <<"  VEnergy  "<<VEnergy<<"  IEnergy  "<<IEnergy << endl;
            if (gradv.Norm() < epsilon)
                break;
        }

        Opers::SetMem(*residualIm, 1.0);
        Opers::Splat(*splatI, *phiinv, *I1, BACKGROUND_STRATEGY_WRAP);
        Opers::Splat(*splatOnes, *phiinv, *residualIm, BACKGROUND_STRATEGY_WRAP);

        delete currV0; currV0 = NULL;
        delete prevV0; prevV0 = NULL;
        delete newV0; newV0 = NULL;
    }








