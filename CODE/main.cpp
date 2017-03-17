/*
CEC2017 Constrained Optimization Test Suite 
Guohua Wu (email: guohuawu@nudt.edu.cn, National University of Defense Technology) 
*/

#include <stdlib.h>    
#include <stdio.h>
#include <math.h>
#include <malloc.h>

void cec17_test_COP(double *x, double *f, double *g,double *h, int nx, int mx,int func_num);

 double *OShift=NULL,*M=NULL,*M1=NULL,*M2=NULL,*y=NULL,*z=NULL,*z1=NULL,*z2=NULL;
 int ini_flag=0,n_flag,func_flag,f5_flag;

int ng_A[28]={1,1,1,2,2,1,1,1,1,1,1,2,3,1,1,1,1,2,2,2,2,3,1,1,1,1,2,2};
int nh_A[28]={1,1,1,1,1,6,2,2,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};	

int  main()
{
	int i,j,k,n,m,func_num;
	double *f=NULL,*x=NULL,*g=NULL,*h=NULL;
	FILE *fpt=NULL;
	char FileName[30];
	m=2; // m denotes the population size.
	n=30; // n denotes the dimension size.
	x=(double *)malloc(m*n*sizeof(double));
	f=(double *)malloc(sizeof(double)  *  m);

	for (i = 0; i < 28; i++)
	{
		func_num=i+1;
		g = (double *)malloc(m*ng_A[func_num-1]*sizeof(double));
		h = (double *)malloc(m*nh_A[func_num-1]*sizeof(double));
		sprintf(FileName, "./inputData/shift_data_%d.txt", func_num);
		fpt = fopen(FileName,"r");
		if (fpt==NULL)
		{
			printf("\n Error: Cannot open input file for reading \n");
		}
		
		if (x==NULL)
			printf("\nError: there is insufficient memory available!\n");

//		for(k=0;k<n;k++)
//		{
//			fscanf(fpt,"%Lf",&x[k]);
//		}
		fclose(fpt);
		for (j = 0; j < n*m; j++)
		{
			x[j]=10.0;
		}
			
		for (k = 0; k < 1; k++)
		{
  		cec17_test_COP(x, f, g,h,n, m,func_num);
			for (j = 0; j < m; j++)
			{
				printf(" f%d(x[%d]) = %Lf,",func_num,j+1,f[j]);
			}
			printf("\n");
		}
		free(g);
		free(h);
		g = NULL;
		h = NULL;
	}
	free(x);
	free(f);
	free(y);
	free(z);
	free(z1);
	free(z2);
	free(M);
	free(M1);
	free(M2);
	free(OShift);
    return 0;
}


