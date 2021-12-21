#ifndef SPKMEANS_H_
#define SPKMEANS_H_

double** readObservationsFile(const char* filepath, char* goal, int* NPointer, int* dPointer);
void spkPython(double** T, int* centsIndices, int N, int k);
double** spkInit(double** obs, int d, int N, int *kPointer);
double** weightAdjMat (double** obs, int d, int N);
double* diagDegMat (double** W, int N);
double* diagDegWrapper (double** W, int N);
void Laplacian (double** W, double* D, int N);
double** jacobiWrapper(double** A, int N);
void printVectorsArray(double** arr, int N, int d);
void printDiagMat(double* vector, int N);

#endif