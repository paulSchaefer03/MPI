#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include "mpi.h"

#define MATRIX_SIZE 800

// Formate für die Matrix-Elemente
#define TYP double
#define FORMAT "%f"
#define IST_INT 0
/* #define TYP int
#define FORMAT "%d" 
#define IST_INT 1 */

void befullenMatrix(TYP *matrix) {
    static int seed_initialized = 0;
    if (!seed_initialized) {
        srand(time(0));
        seed_initialized = 1;
    }
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
#if IST_INT
        matrix[i] = (rand() % 20000001) - 10000000;
#else
        matrix[i] = ((rand() % 20000001) - 10000000) + ((double)rand() / RAND_MAX) * (rand() % 2 == 0 ? 1 : -1);
#endif
    }
}

void multiplizierenMatrix(TYP *matrix_a, TYP *matrix_b, TYP *matrix_c) {
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            matrix_c[i * MATRIX_SIZE + j] = 0;
            for (int k = 0; k < MATRIX_SIZE; k++) {
                matrix_c[i * MATRIX_SIZE + j] += matrix_a[i * MATRIX_SIZE + k] * matrix_b[k * MATRIX_SIZE + j];
            }
        }
    }
}

TYP berechneSummeArray(TYP *array) {
    TYP sum = 0;
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
        sum += array[i];
    }
    return sum;
}

int main(int argc, char *argv[]) {

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Automatisch die passenden Dimensionen berechnen lassen:
    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);

/*     if (dims[0] != dims[1]) {
        if (rank == 0) {
            printf("Fehler: Prozessanzahl erlaubt kein quadratisches Prozessgitter!\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    } */

    int q = dims[0];
/*     if (MATRIX_SIZE % q != 0) {
        if (rank == 0) {
            printf("Fehler: MATRIX_SIZE ist nicht durch q=%d teilbar!\n", q);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
 */
    int block_size = MATRIX_SIZE / q;

    int periods[2] = {1, 1};
    int reorder = 0;
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart_comm);
    MPI_Comm_rank(cart_comm, &rank);
    MPI_Comm_size(cart_comm, &size);
    MPI_Comm_rank(cart_comm, &rank);

    int coords[2];
    MPI_Cart_coords(cart_comm, rank, 2, coords);
    int my_row = coords[0];
    int my_col = coords[1];

    // Datentyp für Submatrix
    MPI_Datatype submatrix_type;
    MPI_Type_vector(block_size, block_size, MATRIX_SIZE, MPI_DOUBLE, &submatrix_type);
    MPI_Type_create_resized(submatrix_type, 0, block_size * sizeof(TYP), &submatrix_type);
    MPI_Type_commit(&submatrix_type);

    // Speicher für lokale Blöcke
    TYP *A_local = malloc(block_size * block_size * sizeof(TYP));
    TYP *B_local = malloc(block_size * block_size * sizeof(TYP));
    TYP *C_local = calloc(block_size * block_size, sizeof(TYP));

    // Scatterv Vorbereitung
    int *sendcounts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));
    for (int i = 0; i < q; i++) {
        for (int j = 0; j < q; j++) {
            sendcounts[i * q + j] = 1;
            displs[i * q + j] = i * MATRIX_SIZE * block_size + j * block_size;
        }
    }

    TYP *matrix_a = NULL;
    TYP *matrix_b = NULL;
    TYP *matrix_c = NULL;

    if (rank == 0) {
        matrix_a = malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(TYP));
        matrix_b = malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(TYP));
        matrix_c = malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(TYP));

        befullenMatrix(matrix_a);
        befullenMatrix(matrix_b);
        multiplizierenMatrix(matrix_a, matrix_b, matrix_c);
        printf("Referenz Ergebnis (seriell): %f\n", berechneSummeArray(matrix_c));
        free(matrix_c);
    }

    MPI_Scatterv(matrix_a, sendcounts, displs, submatrix_type,
                 A_local, block_size * block_size, MPI_DOUBLE, 0, cart_comm);
    MPI_Scatterv(matrix_b, sendcounts, displs, submatrix_type,
                 B_local, block_size * block_size, MPI_DOUBLE, 0, cart_comm);

    // Initial Skew
    for (int i = 0; i < my_row; i++) {
        int src, dest;
        MPI_Cart_shift(cart_comm, 1, -1, &src, &dest);
        MPI_Sendrecv_replace(A_local, block_size * block_size, MPI_DOUBLE, dest, 0, src, 0, cart_comm, MPI_STATUS_IGNORE);
    }

    for (int i = 0; i < my_col; i++) {
        int src, dest;
        MPI_Cart_shift(cart_comm, 0, -1, &src, &dest);
        MPI_Sendrecv_replace(B_local, block_size * block_size, MPI_DOUBLE, dest, 1, src, 1, cart_comm, MPI_STATUS_IGNORE);
    }

    // Hauptschleife von Cannon
    for (int step = 0; step < q; step++) {
        for (int i = 0; i < block_size; i++) {
            for (int j = 0; j < block_size; j++) {
                for (int k = 0; k < block_size; k++) {
                    C_local[i * block_size + j] += A_local[i * block_size + k] * B_local[k * block_size + j];
                }
            }
        }
        int src_a, dest_a;
        MPI_Cart_shift(cart_comm, 1, -1, &src_a, &dest_a);
        MPI_Sendrecv_replace(A_local, block_size * block_size, MPI_DOUBLE, dest_a, 0, src_a, 0, cart_comm, MPI_STATUS_IGNORE);

        int src_b, dest_b;
        MPI_Cart_shift(cart_comm, 0, -1, &src_b, &dest_b);
        MPI_Sendrecv_replace(B_local, block_size * block_size, MPI_DOUBLE, dest_b, 1, src_b, 1, cart_comm, MPI_STATUS_IGNORE);
    }

    // Gatherv Ergebnis
    TYP *C_result = NULL;
    if (rank == 0) {
        C_result = malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(TYP));
    }

    MPI_Gatherv(C_local, block_size * block_size, MPI_DOUBLE,
                C_result, sendcounts, displs, submatrix_type, 0, cart_comm);

    if (rank == 0) {
        printf("Ergebnis (Cannon): %f\n", berechneSummeArray(C_result));
        free(C_result);
        free(matrix_a);
        free(matrix_b);
    }

    free(sendcounts);
    free(displs);
    free(A_local);
    free(B_local);
    free(C_local);
    MPI_Type_free(&submatrix_type);
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}
