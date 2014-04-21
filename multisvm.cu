#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <cuda.h>
#include "helper_cuda_gl.h"
#include "helper_functions.h"
#include "helper_math.h"
#include "helper_string.h"
#include "helper_timer.h"
#include "cublas.h"
#include "src/training/training.cu"
#include "src/testing/testing.cu"
#include "src/common/cuTimer.cu"
#include "src/common/parseinputs.cpp"


void runmulticlassifier(char* ,int ,int ,char* ,int ,int ,int ,float ,float ,int ,float ,float, float, float);

//MultiClass classification using SVM
int main(int argc, char** argv)
{

    int lineNum = 0;
    FILE *fp = NULL;
    fp = fopen("conf.txt","r");

    char strLine[256];
    fgets(strLine,256,fp);
    lineNum=atoi(strLine);

    const char *split = " ";
    char *trainFileName;
    char *testFileName;
    char *featureNum;
    char *trainingSampleNum;
    char *testSampleNum;
    char *classNum;
    char *C;
    char *gamma;	

    for (int i=0; i < lineNum; i++){
        fgets(strLine,256,fp);

        trainFileName=strtok(strLine,split);

        testFileName=strtok(NULL,split);

        featureNum=strtok(NULL,split);

        trainingSampleNum=strtok(NULL,split);

        testSampleNum=strtok(NULL,split);

        classNum=strtok(NULL,split);

        C=strtok(NULL,split);
        gamma=strtok(NULL,split);

        runmulticlassifier(trainFileName,
                atoi(trainingSampleNum),
                atoi(featureNum) ,
                testFileName,
                atoi(testSampleNum),
                1,
                atoi(classNum),
                atoi(C),
                0.001,
                0,
                atof(gamma),
                0,
                0,
                0);
    }

}

/**
 * Runs both training and testing. Provides timings
 * @param trainfilename name of the file containing the training samples
 * @param ntraining number of training samples
 * @param nfeatures number of features in the each training sample
 * @param testfilename name of the file containing the testing samples
 * @param ntesting number of testing samples
 * @param code {0: One vs All, 1: All vs All, 2: Even vs Odd}
 * @param nclasses number of classes in the SVM problem
 * @param ntasks number of binary classification tasks
 * @param C penalization parameter
 * @param tau stopping parameter of the SMO algorithm
 * @param kernelcode type of kernel to use {0 (RBF), 1(linear), 2(polynomial), 3(sigmoid)}
 * @param gamma if using RBF kernel, the value of gamma
 * @param a if using polynomial or sigmoid kernel
 * @param b if using polynomial or sigmoid kernel
 * @param d if using polynomial kernel
 */
void runmulticlassifier(char* trainfilename,
                        int ntraining,
                        int nfeatures,
                        char* testfilename,
                        int ntesting,
                        int code,
                        int nclasses,
                        float C,
                        float tau,
                        int kernelcode,
                        float gamma,
                        float a,
                        float b,
                        float d)
{
    
    for(float gamma=0.1; gamma <100.0 ; gamma*=3 ){
	    cublasStatus status;

	    status = cublasInit();
	    if (status != CUBLAS_STATUS_SUCCESS) {
	        fprintf (stderr, "!!!! CUBLAS initialization error\n");
	    }
	    
	    // Set-Up

		//Allocate memory for xtraindata
		float* h_xtraindata = (float*) malloc(sizeof(float) * ntraining* nfeatures);
		float* h_xtraindatatemp = (float*) malloc(sizeof(float) * ntraining* nfeatures);

		//Allocate memory for xtestdata
		float* h_xtestdata = (float*) malloc(sizeof(float) * ntesting* nfeatures);
		float* h_xtestdatatemp = (float*) malloc(sizeof(float) * ntesting* nfeatures);


		//Allocate memory for traindata_label
		int* h_ytraindata = (int*) malloc(sizeof(int) * ntraining);

		//Allocate memory for testdata_label
		int* h_ytestdata = (int*) malloc(sizeof(int) * ntesting);
		int* h_rdata;
	    
	    memset(h_xtraindata, 0, sizeof(float) * ntraining* nfeatures);
	    memset(h_xtraindatatemp, 0, sizeof(float) * ntraining* nfeatures);
	    memset(h_xtestdata, 0, sizeof(float) * ntesting* nfeatures);
	    memset(h_xtestdatatemp, 0, sizeof(float) * ntesting* nfeatures);

	    memset(h_ytraindata, 0, sizeof(int) * ntraining);
	    memset(h_ytestdata, 0, sizeof(int) * ntesting);
	    
		//Parse data from input file
		printf("Parsing input data...\n");
		parsedatalibsvm(trainfilename, h_xtraindatatemp, h_ytraindata, ntraining, nfeatures, nclasses);
		parsedatalibsvm(testfilename, h_xtestdatatemp, h_ytestdata, ntesting, nfeatures, nclasses);
		//printdata(h_xtestdatatemp, h_ytestdata,  ntesting, nfeatures);
		printf("Parsing input data done!\n");

		int ntasks;

		if( code==0 )
		{
			printf("Code: One Vs All\n");
			ntasks = nclasses;
			//Allocate memory for rdata
		    h_rdata= (int*) malloc(sizeof(int) * nclasses * ntasks);
			generateovacode(h_rdata, nclasses, ntasks);
		}
		else if( code==1 )
		{
			printf("Code All vs All\n");
			ntasks = nclasses*(nclasses-1)/2;
			//Allocate memory for rdata
		    h_rdata= (int*) malloc(sizeof(int) * nclasses * ntasks);
			generateavacode(h_rdata, nclasses, ntasks);
		}
		else if( code==2 )
		{
			printf("Code Odd vs Even\n");
			//Allocate memory for rdata
			ntasks=2;
		    h_rdata= (int*) malloc(sizeof(int) * nclasses * ntasks);
			generateevenoddcode(h_rdata, nclasses, ntasks);
		}

		printcode(h_rdata, nclasses, ntasks);

		float* h_C = (float*) malloc(sizeof(float) * ntasks);
	    for(int i=0; i<ntasks; i++)
	    {
	        h_C[i]=C;
	    }

	    printf("Input Train File Name: %s\n", trainfilename);
	    printf("Input Test File Name: %s\n", testfilename);

		printf("# of training samples: %i\n", ntraining);
		printf("# of testing samples: %i\n", ntesting);
		printf("# of features: %i\n", nfeatures);
		printf("# of tasks: %i\n", ntasks);
		printf("# of classes: %i\n", nclasses);
		printf("Gamma: %f\n", gamma);

		bool iszero=false;

		for (int i=0; i< ntraining; i++)
		{
			for (int j=0; j<nfeatures; j++)
			{   
				//h_xtraindata is the transpose matrix  of h_xtraindatatemp
				h_xtraindata[j*ntraining+i]=h_xtraindatatemp[i*nfeatures+j];
			}
			if(h_ytraindata[i]==0)
			{
				iszero=true;
			}
		}

		for (int i=0; i< ntesting; i++)
		{
			for (int j=0; j<nfeatures; j++)
			{
				//h_xtestdata is the transpose matrix of h_xtestdatatemp
				h_xtestdata[j*ntesting+i]=h_xtestdatatemp[i*nfeatures+j];
			}
		}

		if (iszero)
		{
			for (int i=0; i< ntraining; i++)
			{
				h_ytraindata[i]=h_ytraindata[i]+1;
			}
			for (int i=0; i< ntesting; i++)
			{
				h_ytestdata[i]=h_ytestdata[i]+1;
			}
		}

		free(h_xtraindatatemp);
		free(h_xtestdatatemp);

		int* h_ltesthatdata = (int*) malloc(sizeof(int) * ntesting);

		//Allocate memory for b
		float * h_b= (float*) malloc(sizeof(float) * ntasks);
		for (int i=0; i<ntasks; i++)
		{
			h_b[i]= 0.0f;
		}

		//Allocate memory for adata
		float* h_atraindata= (float*) malloc(sizeof(int) * ntraining * ntasks);

		cuResetTimer();
		float tA1=cuGetTimer();
		
		printf("Training classifier...\n");
		trainclassifier(h_xtraindata,
						h_ytraindata,
						h_rdata,
						h_atraindata,
						ntraining,
						nfeatures,
						nclasses,
						ntasks,
						h_C,
						h_b,
						tau,
						kernelcode,
					    gamma,
						a,
						b,
						d);

		float tA2=cuGetTimer();

		printf("Training classifier done!\n");
		printf("Training time Launch =%.1f usec, finished=%.1f msec\n",tA1*1.e3,tA2);
	    
		for (int j=0; j<ntasks; j++)
		{
			int svnum=0;
			for (int i=0; i<ntraining; i++)
			{
				if(h_atraindata[j*ntraining + i]!=0)
				{
					svnum++;
				}
			}
			printf("Task %i, svnum, %i, b %f\n",j, svnum,h_b[j] );
		}

		int nSV=0;
		for (int i=0; i< ntraining; i++)
		{
			for (int j=0; j< ntasks; j++)
			{
				if(h_atraindata[j*ntraining+i]!=0)
				{
					nSV++;
					break;
				}
			}
		}

		float* h_xtraindatared = (float*) malloc(sizeof(float) * nSV* nfeatures);
		int* h_ltraindatared = (int*) malloc(sizeof(int) * nSV);
		float* h_atraindatared = (float*) malloc(sizeof(float) *ntasks* nSV);

		int k=0;

		for (int i=0; i< ntraining; i++)
		{
			//Check if SV in any tasks
			bool isSV=false;

			for (int j=0; j< ntasks; j++)
			{
				if(h_atraindata[j*ntraining+i]!=0)
				{
					isSV=true;
					break;
				}
			}

			//If SV then copy sample and alphas
			if(isSV)
			{
				for (int j=0; j< ntasks; j++)
				{
					h_atraindatared[j*nSV +k]= h_atraindata[j*ntraining+i];
				}


				for (int j=0; j<nfeatures; j++)
				{
					h_xtraindatared[j*nSV+k]=h_xtraindata[j*ntraining+i];
				}
				h_ltraindatared[k]= h_ytraindata[i];

				k++;
			}
		}


		printf("Testing classifier...\n");

		cuResetTimer();
		float tB1=cuGetTimer();
		testingclassifier(	h_xtraindatared,
							h_xtestdata,
							h_ltraindatared,
							h_ltesthatdata,
							h_rdata,
							h_atraindatared,
							nSV,
							ntesting,
							nfeatures,
							nclasses,
							ntasks,
							h_b,
							gamma,
							a,
							b,
							d,
							kernelcode);

		printf("Testing classifier done\n");
		float tB2=cuGetTimer();
		printf("Testing time Launch =%.1f usec, finished=%.1f msec\n",tB1*1.e3,tB2);

		int errors=0;

		for (int i=0; i<ntesting; i++)
		{
			if( h_ytestdata[i]!=h_ltesthatdata[i])
			{
				errors++;
			}
		}


		printf("%f # of testing samples %i, # errors %i, Rate %f\n",gamma, ntesting, errors, 100* (float) (ntesting -errors)/(float)ntesting);

		free(h_rdata);
		free(h_xtraindata);
		free(h_xtestdata);
		free(h_ytraindata);
		free(h_ytestdata);
		free(h_b);
		free(h_atraindata);
		free(h_xtraindatared);
		free(h_ltraindatared);
		free(h_atraindatared);
    }
} 

