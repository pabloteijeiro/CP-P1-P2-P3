#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int i, done = 0, n, count, total_count;
    double PI25DT = 3.141592653589793238462643;
    double pi, x, y, z;
    int rank, numprocs;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    
    
    while (!done) {
        if (rank == 0) {
            printf("Enter the number of points: (0 quits)\n");
            scanf("%d", &n);
            
            // Distribuir n a todos los procesos
            for (int dest = 1; dest < numprocs; dest++) {
                MPI_Send(&n, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
            }
        } else {
            MPI_Recv(&n, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
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
            
            // Recolección manual de resultados
            if (rank != 0) {
                MPI_Send(&count, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
            } else {
                total_count = count;
                for (int src = 1; src < numprocs; src++) {
                    int temp_count;
                    MPI_Recv(&temp_count, 1, MPI_INT, src, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    total_count += temp_count;
                }
                
                pi = 4.0 * ((double)total_count / (double)n);
                printf("pi is approx. %.16f, Error is %.16f\n", pi, fabs(pi - PI25DT));
            }
        }
    }
    
    MPI_Finalize();
    return 0;
}
