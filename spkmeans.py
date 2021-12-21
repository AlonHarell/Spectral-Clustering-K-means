import numpy as np
import sys
import spkmeans

#####Initialization of random seed#####
np.random.seed(0)

#Given list lst of integers, formats and prints it
def printIntsLst(lst):
    toConvert = [int(lst[i]) for i in range(0, len(lst))]
    string = str(toConvert)
    string = string[1:len(string) - 1].replace(" ", "")
    string = string.replace("'", "")
    print(string)


#####MAIN#####
cmdinput = sys.argv

#Getting arguments:
k = int(cmdinput[1])
goal = cmdinput[2]
filepath = cmdinput[3]

#For any goal, calls the C module, reads observations (or matrix) from file, executes the required computation, and prints.
#If goal != spk, None will be returned. Else, T will be returned.
T = spkmeans.goalsAndSpkOne(k,goal,filepath)

#If goal == spk, T was returned
if (goal == "spk"):
    N = len(T) #N = Number of rows in T
    k = len(T[0]) #If the eigengap herusitic was applied, then should update k. Will be updated from T's number of columns.
    #KMeans++ initialization:
    rand = np.random.choice(N)
    centIndices = [rand]  # centIndices will save the randomly generated indices
    u1 = np.array(T[rand])
    centroids = [u1]
    Z=1
    while (Z != k):
        D = []
        for i in range(0, N):
            minarr = []
            xi = np.array(T[i])
            for j in range(0, Z):
                minarr.append(sum((xi - centroids[j]) ** 2))  # calculate (xi-uj)^2 foreach 0<=j<k and append to the list

            D.append(min(minarr))  # add to D the minimum result
        Z += 1
        Dsum = sum(D)
        P = [D[i] / Dsum for i in range(0, len(D))]  # creates list of probabilities
        rand = np.random.choice(N, p=P)
        centIndices.append(int(rand)) #add chosen centroid indice
        centroids.append(np.array(T[rand]))  # add chosen centroid

    printIntsLst(centIndices)  # Printing indices of centroids chosen
    spkmeans.spkTwo(N,k,T,centIndices) #Calls the rest of the SPK alogrithm, including Kmeans. Centroids printed via C.















