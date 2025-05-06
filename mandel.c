#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>

#define DEBUG 1  // Aseguramos que DEBUG esté activado para imprimir los resultados

#define X_RESN  1024  // resolución en x
#define Y_RESN  1024  // resolución en y

// Límites del conjunto de Mandelbrot
#define X_MIN  -2.0
#define X_MAX   2.0
#define Y_MIN  -2.0
#define Y_MAX   2.0

// Más iteraciones -> imagen más detallada y mayor costo computacional
#define maxIterations  1000

typedef struct complextype {
    float real, imag;
} Compl;

static inline double get_seconds(struct timeval t_ini, struct timeval t_end) {
    return (t_end.tv_usec - t_ini.tv_usec) / 1E6 +
           (t_end.tv_sec - t_ini.tv_sec);
}

int main(int argc, char *argv[]) {
    int rank, size;
    int i, j, k;
    Compl z, c;
    float lengthsq, temp;
    
    // Variables para medición de tiempos
    struct timeval ti_comp, tf_comp, ti_comm, tf_comm;
    double comp_time = 0, comm_time = 0;
    double total_comp_time = 0, total_comm_time = 0;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Calcular filas por proceso
    int rows_per_proc = Y_RESN / size;
    int extra_rows = Y_RESN % size;
    
    // Asignar filas adicionales a los primeros procesos
    int start_row = rank * rows_per_proc + (rank < extra_rows ? rank : extra_rows);
    int end_row = start_row + rows_per_proc + (rank < extra_rows ? 1 : 0);
    int local_rows = end_row - start_row;
    
    // Buffers para los resultados locales y globales
    int *local_res = malloc(local_rows * X_RESN * sizeof(int));
    int *global_res = NULL;
    int *res[Y_RESN]; // Array de punteros para acceso tipo matriz
    
    if (rank == 0) {
        global_res = malloc(Y_RESN * X_RESN * sizeof(int));
        // Inicializar array de punteros para acceso tipo matriz
        for (i = 0; i < Y_RESN; i++) {
            res[i] = global_res + i * X_RESN;
        }
    }
    
    // Arrays para scatterv y gatherv
    int *sendcounts = NULL;
    int *displs = NULL;
    
    if (rank == 0) {
        sendcounts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
        
        int offset = 0;
        for (int p = 0; p < size; p++) {
            int p_rows = rows_per_proc + (p < extra_rows ? 1 : 0);
            sendcounts[p] = p_rows * X_RESN;
            displs[p] = offset;
            offset += sendcounts[p];
        }
    }
    
    // Medir tiempo de computación
    gettimeofday(&ti_comp, NULL);
    
    // Calcular la parte local del conjunto de Mandelbrot
    for (i = 0; i < local_rows; i++) {
        int global_i = start_row + i;
        for (j = 0; j < X_RESN; j++) {
            z.real = z.imag = 0.0;
            c.real = X_MIN + j * (X_MAX - X_MIN) / X_RESN;
            c.imag = Y_MAX - global_i * (Y_MAX - Y_MIN) / Y_RESN;
            k = 0;
            
            do {
                temp = z.real * z.real - z.imag * z.imag + c.real;
                z.imag = 2.0 * z.real * z.imag + c.imag;
                z.real = temp;
                lengthsq = z.real * z.real + z.imag * z.imag;
                k++;
            } while (lengthsq < 4.0 && k < maxIterations);
            
            local_res[i * X_RESN + j] = (k >= maxIterations) ? 0 : k;
        }
    }
    
    gettimeofday(&tf_comp, NULL);
    comp_time = get_seconds(ti_comp, tf_comp);
    
    // Medir tiempo de comunicación
    gettimeofday(&ti_comm, NULL);
    
    // Recolectar resultados en el proceso 0
    MPI_Gatherv(local_res, local_rows * X_RESN, MPI_INT,
                global_res, sendcounts, displs, MPI_INT,
                0, MPI_COMM_WORLD);
    
    gettimeofday(&tf_comm, NULL);
    comm_time = get_seconds(ti_comm, tf_comm);
    
    // Reducir tiempos para obtener totales
    MPI_Reduce(&comp_time, &total_comp_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comm_time, &total_comm_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        fprintf(stderr,"(PERF) Tiempo total de computación (s): %lf\n", total_comp_time);
        fprintf(stderr,"(PERF) Tiempo total de comunicación (s): %lf\n", total_comm_time);
        

        if (DEBUG) {
            for (i = 0; i < Y_RESN; i++) {
                for (j = 0; j < X_RESN; j++) {
                    printf("%3d ", res[i][j]);
                }
                printf("\n");
            }
        }
        
        free(global_res);
        free(sendcounts);
        free(displs);
    }
    
    free(local_res);
    MPI_Finalize();
    
    return 0;
}
