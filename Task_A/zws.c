#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include "mpi.h"

#define MATRIX_SIZE 20

// Formate für die Matrix-Elemente
#define TYP double
#define FORMAT "%f"
#define SUMTYPE_FORMAT "%Lf"
#define IST_INT 0 
#define SUMTYPE long double
/* #define TYP int
#define FORMAT "%d" 
#define SUMTYPE_FORMAT "%lld"
#define IST_INT 1
#define SUMTYPE long long */


void befullenMatrix(TYP *matrix) {
    static int seed_initialized = 0;
    if (!seed_initialized) {
        srand(time(0));
        seed_initialized = 1;
    }
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
        matrix[i] = ((rand() % 200001) - 100000) + ((double)rand() / RAND_MAX);
    }
}

void multiplizierenMatrix(TYP *A, TYP *B, TYP *C) {
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            C[i * MATRIX_SIZE + j] = 0;
            for (int k = 0; k < MATRIX_SIZE; k++) {
                C[i * MATRIX_SIZE + j] += A[i * MATRIX_SIZE + k] * B[k * MATRIX_SIZE + j];
            }
        }
    }
}

SUMTYPE berechenSummeArray(TYP *array) {
    SUMTYPE sum = 0;
    for (int i = 0; i < MATRIX_SIZE*MATRIX_SIZE; i++) {
        sum += array[i];
    }
    return sum;
}

void createSubarrayDatatype(int block_size, MPI_Datatype *newtype, MPI_Datatype base_type) {
    int sizes[2] = {MATRIX_SIZE, MATRIX_SIZE};
    int subsizes[2] = {block_size, block_size};
    int starts[2] = {0, 0};
    MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, base_type, newtype);
    MPI_Type_commit(newtype);
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int q = sqrt(world_size);
    if (q * q != world_size || MATRIX_SIZE % q != 0) {
        if (world_rank == 0) {
            printf("Ungültige Prozessanzahl oder Matrixgröße nicht teilbar!\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int block_size = MATRIX_SIZE / q;
    MPI_Datatype MPI_TYP = IST_INT ? MPI_INT : MPI_DOUBLE;

    // Kartesischer Kommunikator erstellen
    int dims[2] = {q, q}, periods[2] = {1, 1};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);

    int coords[2];
    MPI_Cart_coords(cart_comm, world_rank, 2, coords);

    // Nur Rank 0 hält vollständige Matrizen
    TYP *matrix_a = NULL, *matrix_b = NULL, *matrix_c = NULL;
    if (world_rank == 0) {
        matrix_a = malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(TYP));
        matrix_b = malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(TYP));
        matrix_c = malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(TYP));
        befullenMatrix(matrix_a);
        befullenMatrix(matrix_b);
        multiplizierenMatrix(matrix_a, matrix_b, matrix_c);
        printf("Ergebniss Korrekt:" SUMTYPE_FORMAT "\n", berechenSummeArray(matrix_c));
    }

    // Lokale Blöcke
    TYP *A_local = malloc(block_size * block_size * sizeof(TYP));
    TYP *B_local = malloc(block_size * block_size * sizeof(TYP));
    TYP *C_local = calloc(block_size * block_size, sizeof(TYP));

    // Scatterv vorbereiten
    int *sendcounts = malloc(world_size * sizeof(int));
    int *displs = malloc(world_size * sizeof(int));
    for (int i = 0; i < world_size; i++) {
        sendcounts[i] = 1;
        int row = i / q;
        int col = i % q;
        displs[i] = row * MATRIX_SIZE * block_size + col * block_size;
    }

    MPI_Datatype submatrix_type;
    createSubarrayDatatype(block_size, &submatrix_type, MPI_TYP);

    MPI_Scatterv(matrix_a, sendcounts, displs, submatrix_type, A_local, block_size * block_size, MPI_TYP, 0, MPI_COMM_WORLD);
    MPI_Scatterv(matrix_b, sendcounts, displs, submatrix_type, B_local, block_size * block_size, MPI_TYP, 0, MPI_COMM_WORLD);

    // Initial Skewing
    int src, dst;
    MPI_Cart_shift(cart_comm, 1, -coords[0], &src, &dst);
    MPI_Sendrecv_replace(A_local, block_size * block_size, MPI_TYP, dst, 0, src, 0, cart_comm, MPI_STATUS_IGNORE);
    MPI_Cart_shift(cart_comm, 0, -coords[1], &src, &dst);
    MPI_Sendrecv_replace(B_local, block_size * block_size, MPI_TYP, dst, 0, src, 0, cart_comm, MPI_STATUS_IGNORE);

    for (int step = 0; step < q; step++) {
        // Optimierte Schleife
        for (int idx = 0; idx < block_size * block_size; idx++) {
            int i = idx / block_size;
            int j = idx % block_size;
            for (int k = 0; k < block_size; k++) {
                C_local[i * block_size + j] += A_local[i * block_size + k] * B_local[k * block_size + j];
            }
        }
        if (step < q - 1) {
            MPI_Cart_shift(cart_comm, 1, -1, &src, &dst);
            MPI_Sendrecv_replace(A_local, block_size * block_size, MPI_TYP, dst, 0, src, 0, cart_comm, MPI_STATUS_IGNORE);
            MPI_Cart_shift(cart_comm, 0, -1, &src, &dst);
            MPI_Sendrecv_replace(B_local, block_size * block_size, MPI_TYP, dst, 0, src, 0, cart_comm, MPI_STATUS_IGNORE);
        }
    }

    MPI_Gatherv(C_local, block_size * block_size, MPI_TYP,
                matrix_c, sendcounts, displs, submatrix_type,
                0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        printf("Ergebniss Cannon:" SUMTYPE_FORMAT "\n", berechenSummeArray(matrix_c));
        free(matrix_a); free(matrix_b); free(matrix_c);
    }

    free(A_local); free(B_local); free(C_local);
    free(sendcounts); free(displs);
    MPI_Type_free(&submatrix_type);
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}