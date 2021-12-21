#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include "spkmeans.h"

typedef struct datapoint datapoint;
typedef struct cluster cluster;
typedef struct matCol matCol;

double** readObservationsFile(const char* filepath, char* goal, int* NPointer, int* dPointer);
void spkPython(double** T, int* centsIndices, int N, int k);
void spkC(double** obs, int d, int N, int k);
double** spkInit(double** obs, int d, int N, int *kPointer);
void initDatapointsAndCentroids(cluster *clusters, datapoint *datapoints, double** T, int N, int k);
void kmeans(cluster *clusters, datapoint *datapoints, int N, int d, int K);
double** createT (matCol* M, int N, int k);
matCol* createM (double** obs, int d, int N);
int eigengapHeuristic(matCol* M, int N);
int compareEigenval (const void *elem1, const void *elem2);
double** jacobiWrapper(double** A, int N);
matCol* jacobiAlgorithm (double** A, int N);
void calcAtag(double** A, int N, double c, double s, int i, int j);
double calcOffSquared(double** A, int N);
int* calc_ij(double** A, int N);
double* calc_cs(double** A,int i, int j);
double** identityMat(int N);
void Laplacian (double** W, double* D, int N);
void diagDegSqrt (double* D, int N);
double* diagDegMat (double** W, int N);
double* diagDegWrapper (double** W, int N);
double l2_norm(double*, double*, int);
double** weightAdjMat (double** obs, int d, int N);
int minInt(int i, int j);
int maxInt(int i, int j);
int getVectorSize(char* input, int max_input_size);
double* inputToVector(char* input, int d);
void printCentroids(cluster *clusters, int k);
void printVectorsArray(double** arr, int N, int d);
void printDiagMat(double* vector, int N);
double* getColumn(double** mat, int size, int c);
int minIndex(double v[] , int K);
double* subtractVectors(double v1[], double v2[], int d);
void subtractVectors_inplace(double v1[], double v2[], int d);
void addVectors_inplace(double v1[], double v2[], int d);
double squareVector(double v[], int d);
void divVector(double v[], int a, int d);
double* copyVector(double v[], int d);
int equalVectors(double v1[], double v2[], int d);


struct datapoint /* This struct represents a datapoint and it's cluster */
{
    double* vector;
    int cluster_id; /** current cluster to which datapoint is assigned  */
};

struct cluster /* This struct represents a cluster */
{
    double* centroid;
    double* sum; /** sum of all datapoints in cluster**/
    int count; /** number of datapoints in cluster**/
};

/** matCol is a column in matrix U. It represents an eigenvector(col) and and eigenvalue(eigenVal). 
 * index field is the index of the eigenvalue in the original matrix. */
struct matCol
{
    double* col;
    double eigenVal;
    int index;
};


int main (int argc, char **argv)
{   
    int k, N, i, d;  
    double **observations, **returnedMatrix, *returnedDDG;
    char* goal;
    const char *filepath;
    
    
    if (argc == 42) { /* Supresses warning for unused variable argc. No need to use since no need to validate arguments (Assumption 2.7.12) */}

    k=atoi(argv[1]);
    goal = argv[2];
    filepath = argv[3];

    /* Reads and returns observations matrix from given file. Observations are of size Nxd. N,d updated accordingly (passed by reference). */
    observations = readObservationsFile(filepath, goal, &N, &d);


    /* An if block matching the goal will be executed.
    * The point of an "if-else tree" is to not check the other goals if one is selected, to increase efficiency. */

    /***** goal: wam *****/
    if (strcmp(argv[2], "wam") == 0)
    {
        returnedMatrix = weightAdjMat(observations, d, N); /* Calculates the WAM, observations freed inside */
        printVectorsArray(returnedMatrix, N, N);

        for (i = 0; i < N; i++)
        {
            free(returnedMatrix[i]);
        }
        free(returnedMatrix);
    }
    else
    {

        /***** goal: ddg *****/
        if (strcmp(argv[2], "ddg") == 0)
        {
            returnedMatrix = weightAdjMat(observations, d, N); /* Calculates the WAM, observations freed inside */
            returnedDDG = diagDegMat(returnedMatrix, N);       /* calculates the DDG */
            for (i = 0; i < N; i++)
            {
                free(returnedMatrix[i]);
            }
            free(returnedMatrix);

            /**Printing the DDG.**/
            printDiagMat(returnedDDG, N);
            free(returnedDDG);
        }
        else
        {

            /***** goal: lnorm *****/
            if (strcmp(argv[2], "lnorm") == 0)
            {
                returnedMatrix = weightAdjMat(observations, d, N); /* Calculates the WAM, observations freed inside */
                returnedDDG = diagDegWrapper(returnedMatrix, N);   /* calculates the DDG^(-0.5) */
                Laplacian(returnedMatrix, returnedDDG, N);         /* on-place on returnedMatrix, now returnedMatrix=Lnorm */
                printVectorsArray(returnedMatrix, N, N);

                for (i = 0; i < N; i++)
                {
                    free(returnedMatrix[i]);
                }
                free(returnedMatrix);
                free(returnedDDG);
            }
            else
            {
    
                /***** goal: jacobi *****/
                if (strcmp(argv[2], "jacobi") == 0)
                {
                    /* observations matrix serves as symmetric input matrix in this case */
                    returnedMatrix = jacobiWrapper(observations, N); /*  observations freed inside */
                    printVectorsArray(returnedMatrix, N + 1, N);     /* N+1 since first row is eigenvalues */
                    for (i = 0; i < N+1; i++)
                    {
                        free(returnedMatrix[i]);
                    }
                    free(returnedMatrix);
                }
                else
                {

                    /***** goal: spk *****/
                    if (strcmp(argv[2], "spk") == 0)
                    {
                        spkC(observations, d, N, k);
                    }
                }
            }
        }
    }

    return 0;
}


/** Gets a file address (filepath), goal, and pointers to integers NPointer and dPointer. Reads from the file the observations (or matrix, if goal == jacobi).
 *  Observations matrix is returned, and values of integers to which NPointer and dPointer are updated accordingly:
 * N = Number of observations (or rows in matrix), d = Number of features (or columns in matrix). */
double** readObservationsFile(const char* filepath, char* goal, int* NPointer, int* dPointer)
{
    const int MAX_OBSERVATIONS = 50;
    const int MAX_INPUT_SIZE_NORMAL = 110; /* maximum length of readable line: 10 features, separated by 9 commas each has minus sign, 4 digits before and after decimal point, and 0 character */
    const int MAX_INPUT_SIZE_JACOBI = 550; /* maximum length of readable line: 50 columns, separated by 49 commas, each has minus sign, 4 digits before and after decimal point, and 0 character */
    int N,d = -1, obsSize = 1, MAX_INPUT_SIZE = MAX_INPUT_SIZE_NORMAL;
    char *line; 
    FILE *file;
    double **observations;

    if ((strcmp(goal,"jacobi") == 0)) /** Max size of line in input file varies depending on goal **/
    {
        MAX_INPUT_SIZE = MAX_INPUT_SIZE_JACOBI;
    }

    file = fopen(filepath,"r");
    if (file == NULL) /* In case file cannot be opened - due to either memory error, corruption or other reason */
    {
        printf("An Error Has Occured\n");
        assert(file != NULL);
    }


    N = 0;
    /* allocating space for obsSize pointers (to vectors). Dynamic arrays (doubling) will be used if needed,
     * up to a maximal size of MAX_OBSERVATIONS. */
    observations = (double**)calloc(obsSize, sizeof(double*));
    if (observations == NULL)
    {
        printf("An Error Has Occured\n");
        fclose(file);
        assert(observations != NULL);
    }

    line = (char *)calloc(MAX_INPUT_SIZE, sizeof(char));
    if (line == NULL)
    {
        printf("An Error Has Occured\n");
        fclose(file);
        assert(line != NULL);
    }

    while (fscanf(file, "%s", line) != EOF) /* reading line from input file until EOF reacehd */
    {
        if (d == -1)
        {
            d = getVectorSize(line, MAX_INPUT_SIZE); /* happens once, gets dimension of vectors */
        }
        observations[N] = inputToVector(line, d); /* convert input line to a vector */
        N++;

        free(line);
        line = (char *)calloc(MAX_INPUT_SIZE, sizeof(char));
        if (line == NULL)
        {
            printf("An Error Has Occured\n");
            fclose(file);
            assert(line != NULL);
        }


         /* Array size doubling, O(N) amortized so both efficient in time and memory allocation*/
        if (obsSize == N) /* If obsSize == N then the array is full */
        {

            if (obsSize * 2 >= MAX_OBSERVATIONS) /* If will reach move than max size, make it max size */
            {
                obsSize = MAX_OBSERVATIONS;
            }
            else
            {
                obsSize = obsSize * 2;
            }

            observations = (double**)realloc(observations, obsSize * sizeof(double*)); /* resize observations array to size of obsSize */
            if (observations == NULL)
            {
                printf("An Error Has Occured\n");
                fclose(file);
                assert(observations != NULL);
            }
        }


    }
    fclose(file);
    free(line);

    if (N != obsSize)
    {
        observations = (double**)realloc(observations, N * sizeof(double*)); /* resize observations array to size of N, if needed */
        if (observations == NULL)
        {
            printf("An Error Has Occured\n");
            assert(observations != NULL);
        }
    }

    *NPointer = N;
    *dPointer = d;

    return observations;
}

/* Recieves the matrix T of size Nxk and the indices of chosen centroids, executes K-means (with K-means++ initialization), and prints the centroids.
 * T and centIndices freed inside. */
void spkPython(double** T, int* centsIndices, int N, int k)
{
    int i;
    cluster *clusters;
    datapoint *datapoints;

    datapoints = (datapoint*)calloc(N,sizeof(datapoint)); /* creating an array of N datapoints */
    if (datapoints == NULL)
    {
        printf("An Error Has Occured\n");
        assert(datapoints != NULL);
    }

    clusters = (cluster*)calloc(k,sizeof(cluster)); /* creating an array of k clusters */
    if (clusters == NULL)
    {
        printf("An Error Has Occured\n");
        assert(clusters != NULL);
    }

    /* initializes datapoints and clusters with centroids */
    for (i=0; i<N; i++)
    {
        datapoints[i].cluster_id = -1; /* update datapoint's cluster as none, using sentinel (-1) */
        datapoints[i].vector = T[i]; /* ith datapoint is ith row of T */
    }

    /* Kmeans++ Intilization */
    for (i=0; i<k; i++)
    {
        clusters[i].centroid = copyVector(T[centsIndices[i]],k); /* apply the chosen row from T to be the centroid of the ith cluster */
        clusters[i].sum = (double*)calloc(k,sizeof(double));  /* No vectors assigned, so sum is 0. Initialized to 0s by calloc */
        if (clusters[i].sum == NULL)
        {
            printf("An Error Has Occured\n");
            assert(clusters[i].sum != NULL);
        }
        clusters[i].count = 0;
    }

    free(T);
    free(centsIndices);
    /* the above are no longer used in their own context, therefore can be freed */

    kmeans(clusters,datapoints,N,k,k); /* Kmeans algorithm, updates clusters' centroids */

    /* Freeing datapoints: */
    for (i=0; i<N; i++)
    {
        free(datapoints[i].vector);
    }
    free(datapoints);

    /* PRINTING: */
    printCentroids(clusters,k);

    /* Freeing clusters: */
    for (i=0; i<k; i++)
    {
        free(clusters[i].centroid);
        free(clusters[i].sum);
    }
    free(clusters);


}

/* Spectral Clustering Algoirthm, C implementation (with HW1 initialization). Prints the centroids.
 * Recieves observations of size Nxd and integer k. obs freed inside. */
void spkC(double** obs, int d, int N, int k)
{
    int i;
    double** T;
    cluster *clusters;
    datapoint *datapoints;

    T = spkInit(obs,d,N,&k); /* Returns T, of size Nxk, and updates the variable of k (passed by reference) if needed. obs freed inside. */

    datapoints = (datapoint*)calloc(N,sizeof(datapoint)); /* creating an array of N datapoints */
    if (datapoints == NULL)
    {
        printf("An Error Has Occured\n");
        assert(datapoints != NULL);
    }

    clusters = (cluster*)calloc(k,sizeof(cluster)); /* creating an array of k clusters */
    if (clusters == NULL)
    {
        printf("An Error Has Occured\n");
        assert(clusters != NULL);
    }

    initDatapointsAndCentroids(clusters,datapoints,T,N,k); /* HW1 initialization of the datapoints and clusters. */
    free(T); /* T's rows are used as datapoints, freed later. T itself can be freed. */

    kmeans(clusters,datapoints,N,k,k);  /* Kmeans algorithm, updates clusters' centroids */

    /* Freeing datapoints: */
    for (i=0; i<N; i++)
    {
        free(datapoints[i].vector);
    }
    free(datapoints);

    /* PRINTING: */
    printCentroids(clusters,k);

    /* Freeing clusters: */
    for (i=0; i<k; i++)
    {
        free(clusters[i].centroid);
        free(clusters[i].sum);
    }
    free(clusters);

}

/* Creates and returns the matrix T. Recives observations of size Nxd, and a pointer to integer k.
 * Updates k with eigengap heuristic if k == 0. obs freed inside. */
double** spkInit(double** obs, int d, int N, int *kPointer)
{
    double** T;
    matCol* M;
    int k = *kPointer;
    M = createM(obs, d, N); /* Creates a MatCol array of eigenvectors and eigenvalues */
    qsort(M,N,sizeof(matCol),compareEigenval); /* Sorts M's columns in-place by it's eignvalues */

    if (k == 0)
    {
        k = eigengapHeuristic(M,N);
        *kPointer = k;
    }

    T = createT(M,N,k); /* M freed inside */

    return T;
}

/* Intializes datapoints from rows of T, and initializes K-means clusters as in HW1 */
void initDatapointsAndCentroids(cluster *clusters, datapoint *datapoints, double** T, int N, int k)
{
    int i;
    for (i=0; i<N; i++)
    {
        datapoints[i].cluster_id = -1; /* update datapoint's cluster as none (-1) */
        datapoints[i].vector = T[i];

        /* Kmeans Intilization: */
        if (i<k) /* if i<K then datapoint should be assigned to i-th cluster */
        {
            clusters[i].centroid = copyVector(datapoints[i].vector,k);
            clusters[i].sum = copyVector(datapoints[i].vector,k);
            clusters[i].count = 1;
            datapoints[i].cluster_id = i;
        }
        
    }
}

/* The K-means algorithm. Gets an N-sized array of datapoints and K-sized array of clusters with initialized centroids.
 *  Updates the given datapoints and clusters arrays in-place. */
void kmeans(cluster *clusters, datapoint *datapoints, int N, int d, int K)
{
    int max_iter = 300; /* not const since also used as index */
    int i,j, changedCentroid,index_of_min;
    double squareValue, *vectorsDif, *squaresArr, *newcentroid;
    changedCentroid = 1; /* keeps track if centroid changed during iteration */

    squaresArr = (double*)calloc(K,sizeof(double));
    if (squaresArr == NULL)
    {
        printf("An Error Has Occured\n");
        assert(squaresArr != NULL);
    }

    while ((changedCentroid == 1) && (max_iter > 0))
    {
        max_iter -=1;
        changedCentroid = 0;

        for (i=0; i<N; i++) /* iterate over datapoints */
        {
            
            for (j=0; j<K; j++) /* create array of ||datapoint i - centroid j||^2  foreach 0<=j<K */
            {
                vectorsDif = subtractVectors(datapoints[i].vector,clusters[j].centroid,d);
                squareValue = squareVector(vectorsDif,d);
                free(vectorsDif);
                squaresArr[j]=squareValue;
            }
            index_of_min = minIndex(squaresArr,K); /* find index of minimum, meaning closest centroid to datapoint i */


            if ((datapoints[i].cluster_id == -1) || (datapoints[i].cluster_id != index_of_min))  /* datapoint i's destined cluster should change */
            {
                if (datapoints[i].cluster_id != -1) /* datapoint i had a previous cluster (-1 is a sentinel, indicating datapoint isnt assigned to any cluster) */
                {
                    subtractVectors_inplace(clusters[datapoints[i].cluster_id].sum,datapoints[i].vector,d); /* subtract datapoint i from prev cluster's sum */
                    clusters[datapoints[i].cluster_id].count -=1;
                }

                addVectors_inplace(clusters[index_of_min].sum,datapoints[i].vector,d); /* add datapoint i to new cluster's sum */
                clusters[index_of_min].count +=1;
                datapoints[i].cluster_id = index_of_min;
            }

        }

        /* update all centroids: */
        for (i=0; i<K; i++)
        {
            
            newcentroid = copyVector(clusters[i].sum,d);
            if (clusters[i].count != 0)
            {
                divVector(newcentroid,clusters[i].count,d); /* new centroid = sum divided by count */
            }
            /* If 0 datapoints in cluster , sum would be a vector of 0s and thus the centroid will remain as a vector of 0s.
             * While it's noted that this case should not happen, it might happen due to floating point errors. 
             * If input is valid and there are no floating point errors, count == 0 should not happen. */
            
            if (equalVectors(newcentroid,clusters[i].centroid,d) == 0) /* not equal, so should change centroid */
            {
                free(clusters[i].centroid);
                clusters[i].centroid = newcentroid;
                changedCentroid = 1;
            }
            else /* equal, no need to change */
            {
                free(newcentroid);
            }

        }

    }
    free(squaresArr);

}

/* Creates and returns matrix T of size Nxk as per step 5 in the SPK algorithm. Note: M freed inside */
double** createT (matCol *M, int N, int k)
{
    int i,j;
    matCol *U;
    double sumRowU;
    double **T = (double**)calloc(N,sizeof(double*));
    if (T == NULL)
    {
        printf("An Error Has Occured\n");
        assert(T != NULL);
    }

    for (i=k; i<N; i++)
    {
        free(M[i].col);
    }

    M = (matCol*)realloc(M,k*sizeof(matCol));
    U = M; /* Renaming as U for convenience, after unneeded columns were freed. Now U is of size Nxk */

    /* Normalization of U's items, to create T */
    for (i=0; i<N; i++)
    {
        sumRowU = 0;
        for (j=0; j<k; j++)
        {
            sumRowU += pow(U[j].col[i],2);
        }
        sumRowU = sqrt(sumRowU);

        T[i] = (double*)calloc(k,sizeof(double));
        if (T == NULL)
        {
            printf("An Error Has Occured\n");
            assert(T != NULL);
        }

        for (j=0; j<k; j++)
        {
            /* U[j].col[i] is the item in U's ith row and jth column */
            T[i][j] = U[j].col[i] / sumRowU;
        }
    }


    for (i=0; i<k; i++)
    {
        free(U[i].col);
    }
    free(U); /**U=M,  freed here**/

    return T;
}

/* Creates and returns the eigenvectors+values array: each cell is an eigenvalue and the corresponding eigenvector.
 * (creates W, turns it into Lnorm, freed inside jacobiAlogirthm. obs freed inside.) */
matCol* createM (double** obs, int d, int N)
{
    double **WLnorm, *D;
    matCol *M;
    WLnorm = weightAdjMat(obs,d,N); /* currently WLnorm is the Weighted Adjacency matrix */
    D = diagDegWrapper(WLnorm,N); /* creating DDG^(-0.5) */
    Laplacian(WLnorm,D,N); /* currently WLnorm is the Laplacian Lnorm */
    free(D);
    M = jacobiAlgorithm(WLnorm,N);  /* WLnorm is freed inside jacobiAlgorithm! */

    return M;

}

/* The Eigengap Heuristic algorithm, returns k */
int eigengapHeuristic(matCol* M, int N)
{
    int i,k=1, init = 0;
    double maxFound;
    double *deltas = (double*)calloc(N,sizeof(double)); /* cell 0 not used. Code is more on point with the heuristic if starting index is 1. Used for conveniece.*/
    if (deltas == NULL)
    {
        printf("An Error Has Occured\n");
        assert(deltas != NULL);
    }

    /* Calculating deltas: differences between eigenvalues */
    for (i=0; i<N-2; i++)
    {
        deltas[i+1] = fabs(M[i].eigenVal - M[i+1].eigenVal);
    }

    /* Finding the index of the maximum delta among the first floor(N/2)  */
    for (i=1; i<=(int)(floor(N/2)); i++)
    {
        if (init == 0) /* first iteration only */
        {
            maxFound = deltas[i];
            k = i;
            init = 1;
        }
        else
        {
            if (deltas[i] > maxFound)
            {
                maxFound = deltas[i];
                k = i;
            }
        }
    }

    free(deltas);
    return k;

}

/* Comparator method to compare two columns of matrix U by their matching eigenvalue, to be used in qsort.
   Sorting should be stable, so if eigenvalues are equal, sort by index in original matrix */
int compareEigenval (const void *elem1, const void *elem2)
{
    double eigenval1 = ((matCol*)elem1)->eigenVal;
    double eigenval2 = ((matCol*)elem2)->eigenVal;
    int index1 = ((matCol*)elem1)->index;
    int index2 = ((matCol*)elem2)->index;

    if (eigenval1 > eigenval2)
    {
        return 1;
    }

    if (eigenval2 > eigenval1)
    {
        return -1;
    }

    /* case eigenval1 == eigenval2, choose according to index. Enforces stableness of sort */

    if (index1 > index2)
    {
        return 1;
    }
    else /* index2 > index1. (There are never two equal indices) */
    {
        return -1;
    }

}

/** Returns matrix, in which:
 *  1st row is eigenvalues of A. 2nd to N+1th rows are eigenvectors of A.
 * A freed inside. */
double** jacobiWrapper(double** A, int N)
{
    int i,j;
    double** toReturn;
    matCol* jacobi = jacobiAlgorithm(A,N); /* Executes the Jacobi algorithm on matrix A, creating a matCol array */

    toReturn = (double**)calloc(N+1,sizeof(double*)); /* first row of toReturn: eigenvalues. 2nd to N+1th row: eigenvectors */
    if (toReturn == NULL)
    {
        printf("An Error Has Occured\n");
        assert(toReturn != NULL);
    }

    /* Creating the output matrix */
    for (i=0; i<N+1; i++)
    {
        toReturn[i] = (double*)calloc(N,sizeof(double));
        if (toReturn[i] == NULL)
        {
            printf("An Error Has Occured\n");
            assert(toReturn[i] != NULL);
        }

        for (j=0; j<N; j++)
        {
            if (i == 0) /* fill output matrix's first row with eigenvalues */
            {
                toReturn[i][j] = jacobi[j].eigenVal;
            }
            else /* fill output matrix's other rows with eigenvectors */
            {
                toReturn[i][j] = jacobi[i-1].col[j];
            }
        }

        if (i >= 1)
        {
            free(jacobi[i-1].col);
        }
    }
    free(jacobi);

    return toReturn;
}

/* Executes the Jacobi Algorithm for matrix A, returns M:
 * M[i].col is the ith eigenvector of A. M[i].eigenval is the ith eigenvalue of A. 
 * A is freed inside.  */
matCol* jacobiAlgorithm (double** A, int N)
{
    const double epsilon = pow(10,-15);
    double c,s, offA, offAtag, prevValue;
    const int MAX_ITER_JACOBI = 100;
    int A_not_diag = 1, i, j, k, iters = 0, minIndex, maxIndex;
    double **V;
    int* ij;
    double* cs;
    matCol *M = (matCol*)calloc(N,sizeof(matCol));
    if (M == NULL)
    {
        printf("An Error Has Occured\n");
        assert(M != NULL);
    }

    V = identityMat(N); /* Initial V is the identity matrix */

    while ((A_not_diag == 1) && (iters < MAX_ITER_JACOBI))
    {
        iters++;
        ij = calc_ij(A,N); /* finds i,j of largest (of absolute value) off-diag element in A */
        i = ij[0];
        j = ij[1];
        free(ij);

        if (A[i][j] != 0) /*A is alredy diagonal iff A[i][j] = 0 */
        {
            cs = calc_cs(A,i,j); /* calculates c,s */
            c = cs[0];
            s = cs[1];
            free(cs);
        }
        else 
        {
            A_not_diag = 0; /* Loop will end. */
            c = 1;
            s = 0;
            /* If this is iteration 1 (init == 0), P with c=1, s=0 is Identity Matrix, thus V=P=I*/
        }
        

        if (A_not_diag == 1) /* If A is diagonal, P=I so no need to multiply, and algorithm can end because A is diagonal */
        {
            /** This part simulates the multiplication V = V*P in-place and efficiently (O(N)).
                 * Only columns i,j in the matirx V are changed in each iteration. thus, values are calculated accordingly.
                 * The distinction of a minIndex and maxIndex is crucial since P is a-symmetric, and we want to support a search of Aij in both upper and lower triangle. */
            minIndex = minInt(i, j);
            maxIndex = maxInt(i, j);
            for (k = 0; k < N; k++)
            {
                prevValue = V[k][minIndex];
                V[k][minIndex] = c * V[k][minIndex] + (-s) * V[k][maxIndex];
                V[k][maxIndex] = s * prevValue + c * V[k][maxIndex];
            }


            offA = calcOffSquared(A,N);
            calcAtag(A,N,c,s,i,j); /* Converts A into A', in-place */
            offAtag = calcOffSquared(A,N); 

            if (offA - offAtag <= epsilon) /* If true, then convergence reached - A is diagonal, or diagonal enough */
            {
                A_not_diag = 0;
            }

        }
        
    }

    /* Creates the MatCol array: A matrix represented by an array of it's columns (which are the eigenvectors).
     * Each columns has a matching eigenvalue. */
    for (k=0; k<N; k++)
    {
        M[k].col = getColumn(V,N,k);
        M[k].eigenVal = A[k][k];
        M[k].index = k;
    }

    for (k=0; k<N; k++)
    {
        free(V[k]);
        free(A[k]);
    }
    free(V);
    free(A);

    return M;
}

/* Converts matrix A in-place into matrix A', which is the result of stage 1.2.1.6 of jacobi.
 * Receives A, size N, indices i,j of the largest off-diagonal element, and c,s*/
void calcAtag(double** A, int N, double c, double s, int i, int j)
{
    int r;
    double prevAri, prevArj, prevAii;

    for (r=0; r<N; r++)
    {
        if ((r != i) && (r != j))
        {
            prevAri = A[r][i];
            prevArj = A[r][j];
            A[r][i] = c*prevAri - s*prevArj;
            A[r][j] = c*prevArj + s*prevAri;

            /* symmtery: */
            A[i][r] = A[r][i];
            A[j][r] = A[r][j];
        }
        
    }
    
    prevAii = A[i][i];
    A[i][i] = pow(c,2)*prevAii + pow(s,2)*A[j][j] - 2*s*c*A[i][j];
    A[j][j] = pow(s,2)*prevAii + pow(c,2)*A[j][j] + 2*s*c*A[i][j];
    A[i][j] = 0;
    A[j][i] = 0;

}

/* Calculates and returns off(A)^2 */
double calcOffSquared(double** A, int N)
{
    int i,j;
    double frobNorm = 0;
    for (i = 0; i<N; i++)
    {
        for (j = 0; j<N; j++)
        {
            if (i != j)
            {
                frobNorm += pow(A[i][j],2);
            }
        }
    }
    /* Note: off(A)^2 equals sum of squared off-diagonal elements */

    return frobNorm;
}

/* Finds location of maximum absolute-value off-diag element in A.
 * Returns the indices as an array {i,j} */
int* calc_ij(double** A, int N)
{
    int init = 0;
    double max = 0;
    int i = 0, j=0, k, l;
    int* ij = (int*)calloc(2,sizeof(int));
    if (ij == NULL)
    {
        printf("An Error Has Occured\n");
        assert(ij != NULL);
    }

    for (k = 0; k<N; k++)
    {
        for (l=k; l<N; l++) 
        {
            if (l != k) /* Searching upper triangle. A is symmetic, so no need to search lower one. */
            {
                if ((init == 0) || (fabs(A[k][l]) > max))
                {
                    init = 1;
                    max = fabs(A[k][l]);
                    i = k;
                    j = l;
                }
            }
        }
    }
    ij[0]=i;
    ij[1]=j;

    return ij;
}

/* Calculates and retruns values c,s; used in Jacobi algorithm. 
 *  Returned as an array {c,s} */
double* calc_cs(double** A,int i, int j)
{
    double theta,c,s,t;
    int sign;
    double* cs = (double*)calloc(2,sizeof(double));
    if (cs == NULL)
    {
        printf("An Error Has Occured\n");
        assert(cs != NULL);
    }
    /* Calculate theta */
    theta = (A[j][j]-A[i][i])/(2*A[i][j]);

    /* Calculate sign(theta) */
    if (theta >= 0)
    {
        sign = 1;
    }
    else
    {
        sign = -1;
    }

    /* Calculate t, c, s */
    t = sign / (fabs(theta)+sqrt(pow(theta,2) + 1));
    c = 1 / sqrt(pow(t,2) + 1);
    s = t*c;

    cs[0]=c;
    cs[1]=s;

    return cs;

}

/* Returns the identity matrix of size NxN */
double** identityMat(int N)
{
    int j;
    double** I = (double**)calloc(N,sizeof(double*)); 
    if (I == NULL)
    {
        printf("An Error Has Occured\n");
        assert(I != NULL);
    }

    for (j=0; j<N; j++)
    {
        I[j] = (double*)calloc(N,sizeof(double)); /* Initializes all cells to 0 */
        if (I[j] == NULL)
        {
            printf("An Error Has Occured\n");
            assert(I[j] != NULL);
        }

        I[j][j] = 1;
    }

    return I;
}

/* Calculates the Lnorm matrix: I - D^(-0.5) W D^(-0.5).
*  Recieves matrix W, size N, and an array which represents the diagonal of D^(-0.5)
*  Works in-place on W!*/
void Laplacian (double** W, double* D, int N)
{
    int i,j;

    for (i=0; i<N; i++)
    {
        for (j=0; j<N; j++)
        {
            /* Multiplying matrix A by diag matrix D (A*D) affects elements: (AD)_i,j = (A)_i,j * (D)_i,i
             * Multiplying diag matrix D by matrix A (D*A) affects elements: (DA)_i,j = (A)_i,j * (D)_j,j 
             * Thus, (D*A*D)_i,j = (A)_i,j * D_i,i * D_j,j */ 

            W[i][j] = W[i][j] * D[i] * D[j];

            if (i == j) /* We want I - D^(-0.5) W D^(-0.5), so subtract from 1 if on diagonal, else subtract from 0 */
            {
                W[i][j] = 1 - W[i][j];
            }
            else
            {
                W[i][j] = -W[i][j];
            }
        }
    }

}

/* Recives matrix representing the Diagonal Degree Matrix D and applies in-place D^(-0.5) */
void diagDegSqrt (double* D, int N) 
{
    int i;
    for (i=0; i<N; i++)
    {
        D[i] = 1/(sqrt(D[i]));
    }
}

/* Returns the Diagonal Degree Matrix, represented as a 1-dim array of doubles of the diagonal
 * (since only the diagonal is !=0 and relevant) */
double* diagDegMat (double** W, int N)
{
    double sum;
    int i,z;
    double* D = (double*)calloc(N,sizeof(double));
    if (D == NULL)
    {
        printf("An Error Has Occured\n");
        assert(D != NULL);
    }

    /* Calculates the Diagonal Degree Matrix's diagonal, in which each element is the sum of the matching row in W */
    for (i=0; i<N; i++)
    {
        sum = 0;
        for (z=0; z<N; z++)
        {
            sum += W[i][z];
        }

        D[i] = sum;
    }

    return D;
}

/* Creates the Diagonal Degree Matrix D and returns D^(-0.5), represented as a 1-dim array of doubles of the diagonal
 * (since only the diagonal is !=0 and relevant) */
double* diagDegWrapper (double** W, int N)
{
    double* D = diagDegMat(W,N); /* Creates the D*/
    diagDegSqrt(D,N); /* applies D^(-0.5) inplace */

    return D;
}

/* Returns the Weighted Adjacency Matrix of the given N observations(obs) of dimension d.
 * Frees obs inside. */
double** weightAdjMat (double** obs, int d, int N) 
{
    double val;
    int i,j;
    double** W = (double**)calloc(N,sizeof(double*));
    if (W == NULL)
    {
        printf("An Error Has Occured\n");
        assert(W != NULL);
    }

    for (i=0; i<N; i++)
    {
        W[i] = (double*)calloc(N,sizeof(double));
        if (W[i] == NULL)
        {
            printf("An Error Has Occured\n");
            assert(W[i] != NULL);
        }
    }

    for (i=0; i<N; i++)
    {
        for (j=i+1; j<N; j++)
        {
            val = ((-1) * l2_norm(obs[i],obs[j],d))/2; /* Calculates (- ||x_i - x_j ||_2)/2  */
            val = exp(val); /* Calculates e^val */

            /* Assign to both since WAM is symmetic: */
            W[i][j] = val;
            W[j][i] = val;
        }

        /* Note: W[i][i] = 0 since calloc initializes it to 0 */
    }
    

    /*Freeing observations */
    for (i=0; i<N; i++)
    {
        free(obs[i]);
    }
    free(obs);

    return W;
}

/** Returns l2 (Euclidean) norm of vectors v1,v2:  ||v1-v2||_2
 *  Both vectors of dimension d. */
double l2_norm(double* v1, double* v2, int d) 
{
    int i;
    double s = 0;
    double sub;
    for (i = 0; i<d; i++)
    {
        sub = v1[i]-v2[i];
        s += sub*sub;
    }
    return sqrt(s);
}

/** Retruns minimum value of two integers**/
int minInt(int i, int j)
{
    if (i >= j)
    {
        return j;
    }
    return i;
}

/** Retruns maximum value of two integers**/
int maxInt(int i, int j)
{
    if (i >= j)
    {
        return i;
    }
    return j;
}

/* Used for first observation when reading input.
 * Gets a line from the input file (input) of maximum size (max_input_size), and returns number of features. */
int getVectorSize(char* input, int max_input_size)
{
    int i=0, d = 1;
    while ((i<max_input_size) && (input[i] != 0)) /* maximum line size is max_input_size */
    {
        if (input[i] == ',') /* Counting commas to determine number of features (or dimension of matrix) */
        {
            d++;
        }

        i++;
    }
    return d;
}

/* Converts string divided by commas to a vector, returns array of doubles */
double* inputToVector(char* input, int d) 
{
    int i=0;
    double *vector;

    vector = (double*)calloc(d,sizeof(double));
    if (vector == NULL)
    {
        printf("An Error Has Occured\n");
        assert(vector != NULL);
    }

    /*input pointer points to the first char in observation line */
    while (i<d)
    {
        vector[i] = atof(input);  /* build-in atof converts string from pointer until a non-float char (input valid, so ',' or '\0') */
        i++;

        if (i<d) /* if i<d then last element wasnt reached*/
        {
            while (input[0] != ',') /* advance pointer to next element */
            {
                input++;
            }
            input++; /*reached comma, so +1 to get the beginning of next element */
        }
    }
    return vector;
}

/* Prints centroids of given array of k clusters, whose centroids are of dimension k */
void printCentroids(cluster *clusters, int k)
{
    int i,j;
    double toPrint;
    for (i=0; i<k; i++)
    {
        for (j=0; j<k; j++)
        {
            toPrint = clusters[i].centroid[j];
            if ((toPrint <= 0) && (toPrint > -0.00005)) /* Converts -0.0000 to 0.0000 */
            {
                toPrint = 0;
            }
            printf("%.4f",toPrint); 
            if (j < k-1)
            {
                printf(",");
            }
        }
        if (i < k-1)
        {
            printf("\n");
        }
    }
}

/* Prints the matrix arr of size N x d. Values are separated by comma, rows separated by linebreak */
void printVectorsArray(double** arr, int N, int d)
{
    int i,j;
    double toPrint;
    for (i=0; i<N; i++)
    {
        for (j=0; j<d; j++)
        {
            toPrint = arr[i][j];
            if ((toPrint <= 0) && (toPrint > -0.00005)) /* Converts -0.0000 to 0.0000 */
            {
                toPrint = 0;
            }
            printf("%.4f",toPrint);
            if (j < d-1)
            {
                printf(",");
            }
        }

        if (i < N-1)
        {
            printf("\n");
        }
    }
}

/* Prints the given vector as a diagonal matrix of size NxN */
void printDiagMat(double* vector, int N)
{
    int i,j;
    double toPrint;
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            /* Saved as 1-dim array of size N, so value from vector is printed iff i=j, else 0.0000 */
            if (i == j)
            {
                toPrint = vector[i];
                if ((toPrint <= 0) && (toPrint > -0.00005)) /* Converts -0.0000 to 0.0000 */
                {
                    toPrint = 0;
                }
                printf("%.4f", toPrint);
            }
            else
            {
                printf("0.0000");
            }
            if (j < N - 1)
            {
                printf(",");
            }
        }
        if (i < N - 1)
        {
            printf("\n");
        }
    }
}

/* Returns a copy of column c (size passed as argument) from matrix mat */
double* getColumn(double** mat, int size, int c)
{
    int i;
    double* col = (double*)calloc(size, sizeof(double));
    if (col == NULL)
    {
        printf("An Error Has Occured\n");
        assert(col != NULL);
    }

    for (i=0; i<size; i++)
    {
        col[i] = mat[i][c];
    }

    return col;
}

/* Gets v, an array of doubles of size K. Returns index of min element.
 * If min element appears more than once, returns first index of appearance. */
int minIndex(double v[] , int K) 
{
    int i, minIndex;
    double currentMin;
    minIndex = 0;
    currentMin = v[0];
    if (K == 1)
    {
        return 0;
    }

    for (i=1; i<K; i++)
    {
        if (v[i] < currentMin)
        {
            minIndex = i;
            currentMin = v[i];   
        }
    }

    return minIndex;
}

/* Returns new vector v3 = v1 - v2, all of which of dimension d */
double* subtractVectors(double v1[], double v2[], int d) 
{
    int i;
    double *v3;
    v3 = (double*)calloc(d,sizeof(double));
    if (v3 == NULL)
    {
        printf("An Error Has Occured\n");
        assert(v3 != NULL);
    }
    for (i=0; i<d; i++)
    {
        v3[i]=v1[i]-v2[i];
    }

    return v3;
}

/* Foreach i, applies v1[i] = v1[i]-v2[i]. v1,v2 of dimension d */
void subtractVectors_inplace(double v1[], double v2[], int d) 
{
    int i;
    for (i=0; i<d; i++)
    {
        v1[i]=v1[i]-v2[i];
    }
}

/* Foreach i, applies v1[i]=v1[i]+v2[i]. v1,v2 of dimension d*/
void addVectors_inplace(double v1[], double v2[], int d) 
{
    int i;
    for (i=0; i<d; i++)
    {
        v1[i]=v1[i]+v2[i];
    }
}

/* Returns ||v||^2 . v is a vector of dimension d */
double squareVector(double v[], int d) 
{
    int i;
    double s = 0;
    for (i = 0; i<d; i++)
    {
        s += v[i]*v[i];
    }

    return s;
}

/* Divides vector v of dimension d by integer scalar a. Works in-place. */
void divVector(double v[], int a, int d) 
{
    int i;
    for (i=0; i<d; i++)
    {
        v[i] = v[i]/((double)a);
    }
}

/* Returns a copy of vector v (of dimension d) */
double* copyVector(double v[], int d) 
{
    int i;
    double* vcopy;
    vcopy = (double*)calloc(d,sizeof(double));
    if (vcopy == NULL)
    {
        printf("An Error Has Occured\n");
        assert(vcopy != NULL);
    }
    for (i=0; i<d; i++)
    {
        vcopy[i] = v[i];
    }

    return vcopy;
}

/* Returns 1 if vectors v1,v2 are equal, else returns 0. Both v1,v2 are of dimension d. */
int equalVectors(double v1[], double v2[], int d) 
{
    int i;
    for (i=0; i<d; i++)
    {
        if (v1[i] != v2[i])
        {
            return 0;
        }
    }
    return 1;
}

