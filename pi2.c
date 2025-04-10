#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

int MPI_FlattreeColectiva(double *pi, double *partialSolution, int count, 
                         MPI_Datatype datatype, MPI_Op op, int root, 
                         MPI_Comm comm) {
    int rank, numprocs, error;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &numprocs); 

    if(partialSolution == NULL) return MPI_ERR_BUFFER;
    if(count < 0) return MPI_ERR_COUNT;
    if(comm == NULL) return MPI_ERR_COMM;
    if(root < 0 || root >= numprocs) return MPI_ERR_ROOT;
    if(datatype != MPI_DOUBLE) return MPI_ERR_TYPE;

    double recv_pi = *pi;

    error = MPI_Send(&recv_pi, count, datatype, root, 0, comm);
    if(error != MPI_SUCCESS) return error;
    
    if (rank == root) {
        *partialSolution = recv_pi;
        for (int src = 1; src < numprocs; src++) {
            error = MPI_Recv(&recv_pi, count, datatype, src, 0, comm, MPI_STATUS_IGNORE);
            if(error != MPI_SUCCESS) return error;
            *partialSolution += recv_pi;
        }
    }
    
    return error;
}

int MPI_BinomialBcast(void *buffer, int count, MPI_Datatype datatype, 
                     int root, MPI_Comm comm) {
    int rank, numprocs, error;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &numprocs);

    if(buffer == NULL) return MPI_ERR_BUFFER;
    if(count < 0) return MPI_ERR_COUNT;
    if(comm == NULL) return MPI_ERR_COMM;
    if(root < 0 || root >= numprocs) return MPI_ERR_ROOT;
    if(datatype != MPI_INT) return MPI_ERR_TYPE;

    for (int i = 1; i <= ceil(log2(numprocs)); i++) {
        if (rank < pow(2, i)) {
            int paso = pow(2, (i - 1));
            if(rank < paso) {          
                int dest = rank + paso;
                if (dest < numprocs) {
                    error = MPI_Send(buffer, count, datatype, dest, 0, comm);
                    if(error != MPI_SUCCESS) return error;
                }
            }
            else {
                int src = rank - paso;
                error = MPI_Recv(buffer, count, datatype, src, 0, comm, MPI_STATUS_IGNORE);
                if(error != MPI_SUCCESS) return error;
            }
        }
    }
    return error;
}

int main(int argc, char *argv[]) {
    int i, done = 0, n, count, total_count;
    double PI25DT = 3.141592653589793238462643;
    double pi, x, y, z, partialSolution;
    int rank, numprocs, error;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    
    
    while (!done) {
        if (rank == 0) {
            printf("Enter the number of points: (0 quits)\n");
            scanf("%d", &n);
        }
        
        error = MPI_BinomialBcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (error != MPI_SUCCESS) {
            printf("Error in BinomialBcast: %d\n", error);
            MPI_Abort(MPI_COMM_WORLD, error);
        }

        if (n == 0) {
            done = 1;
        } else {
            count = 0;
            
            for (i = rank; i < n; i+= numprocs) {
                x = ((double)rand() + rank) / ((double)RAND_MAX);
                y = ((double)rand() + rank) / ((double)RAND_MAX);
                z = sqrt((x * x) + (y * y));
                if (z <= 1.0) count++;
            }
            
            double count_d = (double)count;
            double total_count_d = 0.0;

            error = MPI_FlattreeColectiva(&count_d, &total_count_d, 1, 
                                        MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (error != MPI_SUCCESS) {
                if (rank == 0) printf("Error in FlattreeColectiva: %d\n", error);
                MPI_Abort(MPI_COMM_WORLD, error);
            }
            if (rank == 0) {
                pi = 4.0 * ((double)total_count_d / (double)n);
                printf("pi is approx. %.16f, Error is %.16f\n", pi, fabs(pi - PI25DT));
            }
        }
    }
    
    MPI_Finalize();
    return 0;
}
