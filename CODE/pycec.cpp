#include <stdlib.h>    
#include <stdio.h>
#include <math.h>
#include <malloc.h>
#include <Python.h>

void cec17_test_COP(double *x, double *f, double *g,double *h, int nx, int mx,int func_num);

double *OShift=NULL,*M=NULL,*M1=NULL,*M2=NULL,*y=NULL,*z=NULL,*z1=NULL,*z2=NULL;
int ini_flag=0,n_flag,func_flag,f5_flag;

int ng_A[28]={1,1,1,2,2,1,1,1,1,1,1,2,3,1,1,1,1,2,2,2,2,3,1,1,1,1,2,2};
int nh_A[28]={1,1,1,1,1,6,2,2,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};	   


static PyObject * cec_solve(PyObject *self, PyObject *args)
{
    double *x=NULL; 
    PyObject * py_x=NULL;
    int nx;
    int mx;
    int func_num;
    double *f=NULL;
    double *g=NULL;
    double *h=NULL;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "O!iii",&PyList_Type ,&py_x, &nx, &mx,
                                        &func_num))
        return NULL;

    f=(double *)malloc(sizeof(double)  *  mx);
	g = (double *)malloc(mx*ng_A[func_num-1]*sizeof(double));
	h = (double *)malloc(mx*nh_A[func_num-1]*sizeof(double));


    int numLines = PyList_Size(py_x);
    if (numLines != mx*nx){
    printf("Wrong population size");
    return NULL;
    }

    x=(double *)malloc(mx*nx*sizeof(double));
    for (int i = 0; i < mx; i++) {

        x[i] =  PyFloat_AS_DOUBLE(PyList_GET_ITEM(py_x,i));   
    }


    cec17_test_COP(x, f, g,h,nx, mx,func_num);
    
    // Build a python list object for f
    PyObject *f_lst = PyList_New(mx);
    if (!f_lst)
        return NULL;
    for (int i = 0; i < mx; i++) {
        PyObject *num = PyFloat_FromDouble(f[i]);
        if (!num) {
            Py_DECREF(f_lst);
        return NULL;
    }
    PyList_SET_ITEM(f_lst, i, num);   // reference to num stolen
    }

    // Build a python list object for g
    PyObject *g_lst = PyList_New(mx*ng_A[func_num-1]);
    if (!g_lst)
        return NULL;
    for (int i = 0; i < mx; i++) {
        PyObject *num = PyFloat_FromDouble(g[i]);
        if (!num) {
            Py_DECREF(g_lst);
        return NULL;
    }
    PyList_SET_ITEM(g_lst, i, num);   // reference to num stolen
    }

    // Build a python list object for h
    PyObject *h_lst = PyList_New(mx*nh_A[func_num-1]);
    if (!h_lst)
        return NULL;
    for (int i = 0; i < mx; i++) {
        PyObject *num = PyFloat_FromDouble(h[i]);
        if (!num) {
            Py_DECREF(h_lst);
        return NULL;
    }
    PyList_SET_ITEM(h_lst, i, num);   // reference to num stolen
    }
    // Free memory
    free(x);
    free(OShift);
    free(M);
    free(M1);
    free(y);
    free(z);
    free(z1);
    free(z2);



    return Py_BuildValue("[OOO]", f_lst,g_lst ,h_lst);

}



static PyMethodDef CECMethods[] = {
 { "cec",cec_solve , METH_VARARGS, "cec 2017 benchmarks" },
 { NULL, NULL, 0, NULL }
};

PyMODINIT_FUNC initcec(void)
{
    PyObject *m = Py_InitModule3("cec", CECMethods, "CEC 2017 benchmarks");
    if (m == NULL)
        return;

}


