
#ifndef _REDUCTION_KERNEL_H_
#define _REDUCTION_KERNEL_H_

#include <stdio.h>


/**
 * Performs an optimized local reduction step to find Iup, Ilow, Bup and Blow
 * @param d_ytraindata device pointer to the array of binary labels
 * @param d_atraindata device pointer to the array of alphas
 * @param d_fdata device pointer to the array of fs
 * @param d_bup device pointer to the local bup values
 * @param d_blow device pointer to the local blow values
 * @param d_Iup device pointer to the local Iup values
 * @param d_Ilow device pointer to the local Ilow values
 * @param d_done_device pointer to the array with the status of each binary task
 * @param d_active device pointer to the array with active binary tasks
 * @param ntraining number of training samples in the training set
 * @param ntasks number of binary tasks to be solved
 * @param activeTasks number of active tasks
 * @param d_C device pointer to the array of regularization parameters
 */
template <unsigned int blockSize, bool isNtrainingPow2>
__global__ static void reduction(			int* d_ytraindata,
											float* d_atraindata,
											float* d_fdata,
											float* d_bup,
											float* d_blow,
											int* d_Iup,
											int* d_Ilow,
											int* d_done,
											int* d_active,
											int ntraining,
											int ntasks,
											int activeTasks,
											float* d_C)
{
	const unsigned int tid = threadIdx.x;
	const unsigned int bidx = blockIdx.x;
	unsigned int j = blockIdx.y;
	unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;
	unsigned int gridSize = blockSize*2*gridDim.x;

	int bidy= d_active[j];

	if(d_done[bidy]== 0)
	{
		float C= d_C[bidy];

		__shared__ float minreduction [blockSize];
		__shared__ float maxreduction [blockSize];
		__shared__ int minreductionid [blockSize];
		__shared__ int maxreductionid [blockSize];

		//Each thread loads one element
		minreduction[tid]= (float)FLT_MAX;
		maxreduction[tid]= -1.0* (float)FLT_MAX;

		minreductionid[tid]= i;
		maxreductionid[tid]= i;

		while (i < ntraining)
		{

				float alpha_i;
				int y_i= d_ytraindata[bidy* ntraining + i];
				float minval=(float)FLT_MAX;
				float maxval= -1.0* (float)FLT_MAX;
				float f_i= d_fdata[bidy* ntraining +i];


				if(y_i !=0)
				{
					alpha_i= d_atraindata[bidy* ntraining + i];

					if(((y_i==1 && alpha_i>0 && alpha_i<C) || (y_i== -1 && alpha_i>0 && alpha_i<C)) ||  (y_i==1 && alpha_i==0) || (y_i== -1 && alpha_i==C) )
					{
						minval=f_i;
					}

					if(((y_i==1 && alpha_i>0 && alpha_i<C) || (y_i== -1 && alpha_i>0 && alpha_i<C)) ||  (y_i==1 && alpha_i==C) || (y_i== -1 && alpha_i==0))
					{
						maxval= f_i;
					}


					if(minreduction[tid] > minval)
					{
						minreduction[tid]= minval;
						minreductionid[tid]= i;
					}

					if(maxreduction[tid] < maxval)
					{
						maxreduction[tid]= maxval;
						maxreductionid[tid]= i;
					}

				}



				if (isNtrainingPow2 || i + blockSize < ntraining)
				{
					y_i= d_ytraindata[bidy* ntraining + i + blockSize];
					minval=(float)FLT_MAX;
					maxval= -1.0* (float)FLT_MAX;
					f_i=d_fdata[bidy* ntraining +i + blockSize];


					if(y_i != 0)
					{
						alpha_i= d_atraindata[bidy* ntraining + i + blockSize];

						if(((y_i==1 && alpha_i>0 && alpha_i<C) || (y_i== -1 && alpha_i>0 && alpha_i<C)) ||  (y_i==1 && alpha_i==0) || (y_i== -1 && alpha_i==C) )
						{
							minval=f_i;


						}

						if(((y_i==1 && alpha_i>0 && alpha_i<C) || (y_i== -1 && alpha_i>0 && alpha_i<C)) ||  (y_i==1 && alpha_i==C) || (y_i== -1 && alpha_i==0))
						{
							maxval= f_i;


						}


						if(minreduction[tid] > minval)
						{
							minreduction[tid]= minval;
							minreductionid[tid]= i + blockSize;
						}

						if(maxreduction[tid] < maxval)
						{
							maxreduction[tid]= maxval;
							maxreductionid[tid]= i + blockSize;
						}
					}

				}

			 i += gridSize;
		 }


		__syncthreads();


		if(blockSize>=512)
		{
			if(tid<256)
			{
				if(minreduction[tid] > minreduction[tid+256])
				{
					minreduction[tid]= minreduction[tid +256];
					minreductionid[tid]= minreductionid[tid +256];
				}

				if(maxreduction[tid] < maxreduction[tid+256])
				{
					maxreduction[tid]= maxreduction[tid +256];
					maxreductionid[tid]= maxreductionid[tid +256];
				}
			}
			__syncthreads();
		}

		if(blockSize>=256)
		{
			if(tid<128)
			{
				if(minreduction[tid] > minreduction[tid+128])
				{
					minreduction[tid]= minreduction[tid +128];
					minreductionid[tid]= minreductionid[tid +128];
				}

				if(maxreduction[tid] < maxreduction[tid+128])
				{
					maxreduction[tid]= maxreduction[tid +128];
					maxreductionid[tid]= maxreductionid[tid +128];
				}
			}
			__syncthreads();
		}

		if(blockSize>=128)
		{
			if(tid<64)
			{
				if(minreduction[tid] > minreduction[tid+64])
				{
					minreduction[tid]= minreduction[tid +64];
					minreductionid[tid]= minreductionid[tid +64];
				}

				if(maxreduction[tid] < maxreduction[tid+64])
				{
					maxreduction[tid]= maxreduction[tid +64];
					maxreductionid[tid]= maxreductionid[tid +64];
				}
			}
			__syncthreads();
		}

		if(tid<32)
		{
			//32
			if(blockSize >= 64)
			{

				if(minreduction[tid] > minreduction[tid+32])
				{
					minreduction[tid]= minreduction[tid +32];
					minreductionid[tid]= minreductionid[tid +32];
				}

				if(maxreduction[tid] < maxreduction[tid+32])
				{
					maxreduction[tid]= maxreduction[tid +32];
					maxreductionid[tid]= maxreductionid[tid +32];
				}
			}
			//16
			if(blockSize >= 32)
			{
				if(minreduction[tid] > minreduction[tid+16])
				{
					minreduction[tid]= minreduction[tid +16];
					minreductionid[tid]= minreductionid[tid +16];
				}

				if(maxreduction[tid] < maxreduction[tid+16])
				{
					maxreduction[tid]= maxreduction[tid +16];
					maxreductionid[tid]= maxreductionid[tid +16];
				}
			}
			//8
			if(blockSize >= 16)
			{
				if(minreduction[tid] > minreduction[tid+8])
				{
					minreduction[tid]= minreduction[tid +8];
					minreductionid[tid]= minreductionid[tid +8];
				}

				if(maxreduction[tid] < maxreduction[tid+8])
				{
					maxreduction[tid]= maxreduction[tid +8];
					maxreductionid[tid]= maxreductionid[tid +8];
				}
			}
			//4
			if(blockSize >= 8)
			{
				if(minreduction[tid] > minreduction[tid+4])
				{
					minreduction[tid]= minreduction[tid +4];
					minreductionid[tid]= minreductionid[tid +4];
				}

				if(maxreduction[tid] < maxreduction[tid+4])
				{
					maxreduction[tid]= maxreduction[tid +4];
					maxreductionid[tid]= maxreductionid[tid +4];
				}
			}
			//2
			if(blockSize >= 4)
			{
				if(minreduction[tid] > minreduction[tid+2])
				{
					minreduction[tid]= minreduction[tid +2];
					minreductionid[tid]= minreductionid[tid +2];
				}

				if(maxreduction[tid] < maxreduction[tid+2])
				{
					maxreduction[tid]= maxreduction[tid +2];
					maxreductionid[tid]= maxreductionid[tid +2];
				}
			}

			//1
			if(blockSize >= 2)
			{
				if(minreduction[tid] > minreduction[tid+1])
				{
					minreduction[tid]= minreduction[tid +1];
					minreductionid[tid]= minreductionid[tid +1];
				}

				if(maxreduction[tid] < maxreduction[tid+1])
				{
					maxreduction[tid]= maxreduction[tid +1];
					maxreductionid[tid]= maxreductionid[tid +1];
				}
			}
		}

		if(tid==0)
		{
			d_bup[bidy * gridDim.x + bidx]=minreduction[tid];
			d_blow[bidy * gridDim.x + bidx]=maxreduction[tid];
			d_Iup[bidy * gridDim.x + bidx]= minreductionid[tid];
			d_Ilow[bidy * gridDim.x + bidx]= maxreductionid[tid];

		}

	}

}


/**
 * Performs an optimized local global reduction to find global Iup, Ilow, Bup and Blow
 * @param d_bup device pointer to the local bup values
 * @param d_blow device pointer to the local blow values
 * @param d_Iup device pointer to the local Iup values
 * @param d_Ilow device pointer to the local Ilow values
 * @param d_done_device pointer to the array with the status of each binary task
 * @param d_active device pointer to the array with active binary tasks
 * @param n number of local blockwise reduction results that need to be globally reduced
 * @param activeTasks number of active tasks
 */
template <unsigned int blockSize, bool isNtrainingPow2>
__global__ static void globalreduction(		float* d_bup,
												float* d_blow,
												int* d_Iup,
												int* d_Ilow,
												int* d_done,
												int* d_active,
												int n,
												int activeTasks)
{
		const unsigned int tid = threadIdx.x;
		const unsigned int bidx = blockIdx.x;
		unsigned int j = blockIdx.y;
		unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;
		unsigned int gridSize = blockSize*2*gridDim.x;

		int bidy= d_active[j];

		//Check if the task has converged
		if(d_done[bidy]== 0)
		{

			__shared__ float minreduction [blockSize];
			__shared__ float maxreduction [blockSize];
			__shared__ int minreductionid [blockSize];
			__shared__ int maxreductionid [blockSize];

			//Each thread loads one element
			minreduction[tid]= (float)FLT_MAX;
			maxreduction[tid]= (float)FLT_MIN;
			minreductionid[tid]= i;
			maxreductionid[tid]= i;

			while (i < n)
			{

				float minval=d_bup[bidy* n + i];
				float maxval= d_blow[bidy* n + i];

				if(minreduction[tid] > minval)
				{
					minreduction[tid]= minval;
					minreductionid[tid]= d_Iup[bidy* n + i];
				}

				if(maxreduction[tid] < maxval)
				{
					maxreduction[tid]= maxval;
					maxreductionid[tid]= d_Ilow[bidy* n + i];
				}

				if (isNtrainingPow2 || i + blockSize < n)
				{
					minval=d_bup[bidy* n + i+blockSize];
					maxval= d_blow[bidy* n + i+blockSize];

					if(minreduction[tid] > minval)
					{
						minreduction[tid]= minval;
						minreductionid[tid]= d_Iup[bidy* n + i+blockSize];
					}

					if(maxreduction[tid] < maxval)
					{
						maxreduction[tid]= maxval;
						maxreductionid[tid]= d_Ilow[bidy* n + i+blockSize];
					}
				}

				i += gridSize;

			}


			__syncthreads();


			if(blockSize>=512)
			{
				if(tid<256)
				{
					if(minreduction[tid] > minreduction[tid+256])
					{
						minreduction[tid]= minreduction[tid +256];
						minreductionid[tid]= minreductionid[tid +256];
					}

					if(maxreduction[tid] < maxreduction[tid+256])
					{
						maxreduction[tid]= maxreduction[tid +256];
						maxreductionid[tid]= maxreductionid[tid +256];
					}
				}
				__syncthreads();
			}

			if(blockSize>=256)
			{
				if(tid<128)
				{
					if(minreduction[tid] > minreduction[tid+128])
					{
						minreduction[tid]= minreduction[tid +128];
						minreductionid[tid]= minreductionid[tid +128];
					}

					if(maxreduction[tid] < maxreduction[tid+128])
					{
						maxreduction[tid]= maxreduction[tid +128];
						maxreductionid[tid]= maxreductionid[tid +128];
					}
				}
				__syncthreads();
			}

			if(blockSize>=128)
			{
				if(tid<64)
				{
					if(minreduction[tid] > minreduction[tid+64])
					{
						minreduction[tid]= minreduction[tid +64];
						minreductionid[tid]= minreductionid[tid +64];
					}

					if(maxreduction[tid] < maxreduction[tid+64])
					{
						maxreduction[tid]= maxreduction[tid +64];
						maxreductionid[tid]= maxreductionid[tid +64];
					}
				}
				__syncthreads();
			}

			if(tid<32)
			{


				//32
				if(blockSize >= 64)
				{

					if(minreduction[tid] > minreduction[tid+32])
					{
						minreduction[tid]= minreduction[tid +32];
						minreductionid[tid]= minreductionid[tid +32];
					}

					if(maxreduction[tid] < maxreduction[tid+32])
					{
						maxreduction[tid]= maxreduction[tid +32];
						maxreductionid[tid]= maxreductionid[tid +32];
					}
				}
				//16
				if(blockSize >= 32)
				{
					if(minreduction[tid] > minreduction[tid+16])
					{
						minreduction[tid]= minreduction[tid +16];
						minreductionid[tid]= minreductionid[tid +16];
					}

					if(maxreduction[tid] < maxreduction[tid+16])
					{
						maxreduction[tid]= maxreduction[tid +16];
						maxreductionid[tid]= maxreductionid[tid +16];
					}
				}
				//8
				if(blockSize >= 16)
				{
					if(minreduction[tid] > minreduction[tid+8])
					{
						minreduction[tid]= minreduction[tid +8];
						minreductionid[tid]= minreductionid[tid +8];
					}

					if(maxreduction[tid] < maxreduction[tid+8])
					{
						maxreduction[tid]= maxreduction[tid +8];
						maxreductionid[tid]= maxreductionid[tid +8];
					}
				}
				//4
				if(blockSize >= 8)
				{
					if(minreduction[tid] > minreduction[tid+4])
					{
						minreduction[tid]= minreduction[tid +4];
						minreductionid[tid]= minreductionid[tid +4];
					}

					if(maxreduction[tid] < maxreduction[tid+4])
					{
						maxreduction[tid]= maxreduction[tid +4];
						maxreductionid[tid]= maxreductionid[tid +4];
					}
				}
				//2
				if(blockSize >= 4)
				{
					if(minreduction[tid] > minreduction[tid+2])
					{
						minreduction[tid]= minreduction[tid +2];
						minreductionid[tid]= minreductionid[tid +2];
					}

					if(maxreduction[tid] < maxreduction[tid+2])
					{
						maxreduction[tid]= maxreduction[tid +2];
						maxreductionid[tid]= maxreductionid[tid +2];
					}
				}

				//1
				if(blockSize >= 2)
				{
					if(minreduction[tid] > minreduction[tid+1])
					{
						minreduction[tid]= minreduction[tid +1];
						minreductionid[tid]= minreductionid[tid +1];
					}

					if(maxreduction[tid] < maxreduction[tid+1])
					{
						maxreduction[tid]= maxreduction[tid +1];
						maxreductionid[tid]= maxreductionid[tid +1];
					}
				}
			}


			if(tid==0)
			{
				d_bup[bidy * gridDim.x + bidx]=minreduction[tid];
				d_blow[bidy * gridDim.x + bidx]=maxreduction[tid];
				d_Iup[bidy * gridDim.x + bidx]= minreductionid[tid];
				d_Ilow[bidy * gridDim.x + bidx]= maxreductionid[tid];

			}

		}
		__syncthreads();
}

#endif // _REDUCTION_KERNEL_H_
