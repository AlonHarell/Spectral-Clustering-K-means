#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "spkmeans.h"


static PyObject* goalsAndSpkOne(PyObject *self, PyObject *args);
static PyObject* spkTwo(PyObject *self, PyObject *args);
static double** PyList_ToMat(PyObject *pyLst, int N, int d);
static PyObject* Mat_ToPyList(double** mat, int N, int d);

/* Activates for all goals and phase one of SPK : steps 1 to 5 in the algorithm.
 * If goal != spk, does the required calculations and prints the result.
 * If goal == spk, returns T of size Nxk. */
static PyObject* goalsAndSpkOne(PyObject *self, PyObject *args)
{
    int i,N,d,k;
    char *goal;
    const char *filepath;
    double **observations, **mat;
    double *diag;
    PyObject *toReturn=Py_None; /* toReturn only needed for SPK so default value is None */
    if(!PyArg_ParseTuple(args, "iss",&k,&goal,&filepath)) /* Parses arguments from python to c*/
    {
        return NULL;
    }

    /* Reads and returns observations matrix from given file. Observations are of size Nxd. N,d updated accordingly (passed by reference). */
    observations = readObservationsFile(filepath,goal,&N,&d);

    /* An if block matching the goal will be executed.
     * The point of an "if-else tree" is to not check the other goals if one is selected, to increase efficiency. */

    if (strcmp(goal,"wam") == 0)
    {
        /***** goal: wam *****/
        mat = weightAdjMat(observations,d,N); /** observations freed inside **/
        printVectorsArray(mat,N,N); /**Printing the WAM**/
        for (i=0; i<N; i++)
        {
            free(mat[i]);
        }
        free(mat);

    }
    else
    {
        /***** goal: ddg *****/
        if (strcmp(goal,"ddg") == 0)
        {
            mat = weightAdjMat(observations, d, N); /** observations freed inside **/
            diag = diagDegMat(mat, N);              /**calculates the DDG**/
            for (i = 0; i < N; i++)
            {
                free(mat[i]);
            }
            free(mat);

            printDiagMat(diag, N); /** Printing the DDG **/
            free(diag);
        }
        else
        {
            /***** goal: lnorm *****/
            if (strcmp(goal, "lnorm") == 0)
            {
                mat = weightAdjMat(observations, d, N); /** observations freed inside **/
                diag = diagDegWrapper(mat, N);          /**calculates the DDG^(-0.5)**/
                Laplacian(mat, diag, N);                /**on-place on returnedMatrix, now returnedMatrix=Lnorm**/
                free(diag);

                printVectorsArray(mat, N, N); /** Printing Lnorm **/

                for (i = 0; i < N; i++)
                {
                    free(mat[i]);
                }
                free(mat);
            }
            else
            {
                /***** goal: jacobi *****/
                if (strcmp(goal,"jacobi") == 0)
                {
                    mat = jacobiWrapper(observations, N); /** observations freed inside **/
                    printVectorsArray(mat, N + 1, N);     /**Printing the returned eigenvectors and eigenvalues. N+1 since first row is eigenvalues**/

                    for (i = 0; i < N; i++)
                    {
                        free(mat[i]);
                    }
                    free(mat);
                }
                else
                {
                    /***** goal: spk *****/
                    if (strcmp(goal,"spk") == 0)
                    {
                        mat = spkInit(observations, d, N, &k); /* observations freed inside. Returned T, and k updated if needed(since passed by-reference)*/
                        toReturn = Mat_ToPyList(mat, N, k);    /* mat freed inside! converting mat to a python returnable */
                    }
                }
            }
        }
    }

    return Py_BuildValue("O",toReturn); /* If goal = spk, T will be returned. Else, Py_None */
}

/* Activates for phase two of SPK : step 6 of the algorithm, after centroids were chosen in python.
 * Executes the Kmeans algorithm (With Kmeans++ initialization) and prints the centroids. */
static PyObject* spkTwo(PyObject *self, PyObject *args)
{
    int i,N,k;
    PyObject *datapointsPy, *centIndicesPy;
    double **datapoints;
    int *centIndices;
    if(!PyArg_ParseTuple(args, "iiOO", &N,&k,&datapointsPy,&centIndicesPy)) /* Parses arguments from Python to C */
    {
        return NULL;
    }

    /* Turning the list of centroid indices (indices of rows in T) into a C array */
    centIndices = (int*)calloc(k,sizeof(int));
    if (centIndices == NULL)
    {
        printf("An Error Has Occured\n");
        assert(centIndices != NULL);
    }
    for (i=0; i<k; i++)
    {
        centIndices[i] = (int)PyFloat_AsDouble(PyList_GetItem(centIndicesPy,i));
    }

    datapoints = PyList_ToMat(datapointsPy, N, k);

    /* Phase 2 of the SPK algorithm. datapoints and centIndices freed inside. Prints centroids. */
    spkPython(datapoints, centIndices,N,k); 

    return Py_BuildValue("O",Py_None); /* Returning None */
}

/* Converts a PyObject (which represents a list of lists) to an N x d matrix of doubles, which is returned. */
static double** PyList_ToMat(PyObject *pyLst, int N, int d)
{
    int i,j;
    PyObject *pyVector;
    double** mat = (double**)calloc(N,sizeof(double*));
    if (mat == NULL)
    {
        printf("An Error Has Occured\n");
        assert(mat != NULL);
    }

    for (i=0; i<N; i++)
    {
        mat[i] = (double*)calloc(d,sizeof(double));
        if (mat[i] == NULL)
        {
            printf("An Error Has Occured\n");
            assert(mat[i] != NULL);
        }

        pyVector = PyList_GetItem(pyLst,i); /* temp vector, points to ith inner list */
        for (j=0; j<d; j++)
        {
            mat[i][j] = PyFloat_AsDouble(PyList_GetItem(pyVector,j));
        }
    }

    return mat;
}

/* Converts an N x d doubles matrix to a PyObject (represents a list of lists), which is returned.  mat freed inside. */
static PyObject* Mat_ToPyList(double** mat, int N, int d)
{
    int i,j;
    PyObject *pyLst, *pyVector;
    pyLst = PyList_New(N);

    for (i=0; i<N; i++)
    {
        pyVector = PyList_New(d); /* building the ith row as a list of floats of len N */
        for (j=0; j<d; j++)
        {
            PyList_SetItem(pyVector,j,PyFloat_FromDouble(mat[i][j]));
        }
        PyList_SetItem(pyLst,i,pyVector);
        free(mat[i]);
    }
    free(mat);

    return pyLst;
}

static PyMethodDef spkMethods[] = {
    {"goalsAndSpkOne",(PyCFunction)goalsAndSpkOne,METH_VARARGS,PyDoc_STR("Activates for all goals and phase one of SPK : steps 1 to 5 in the algorithm. Reads observations from given filepath. Arguments: k, goal, filepath")},
    {"spkTwo",(PyCFunction)spkTwo,METH_VARARGS,PyDoc_STR("Activates for phase two of SPK : step 6 of the algorithm, after centroids were chosen in python. Arguments: N, k, T, centroid indices")},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef moduledef = { PyModuleDef_HEAD_INIT, "spkmeans", NULL, -1, spkMethods};


PyMODINIT_FUNC PyInit_spkmeans(void)
{
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }
    return m;
}