#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include "mpi.h"

#define MATRIX_SIZE 800

// Formate für die Matrix-Elemente
/* #define TYP double
#define FORMAT "%f"
#define IST_INT 0 */
#define TYP int
#define FORMAT "%d" 
#define IST_INT 1


void ausgabeArrayAlsMatrix(TYP* arr) {
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            printf(FORMAT " ", arr[i * MATRIX_SIZE + j]);
        }
        printf("\n");
    }
    printf("\n");
}

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


TYP berechenSummeArray(TYP *array) {
    TYP sum = 0;
    for (int i = 0; i < MATRIX_SIZE*MATRIX_SIZE; i++) {
        sum += array[i];
    }
    return sum;
}


int main(int argc, char *argv[]) {

    int rank, size, ierr;

    MPI_Status status;
    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int q = sqrt(size);  // Prozessgittergröße
    if (MATRIX_SIZE % q != 0) {
        if (rank == 0) {
            printf("Matrixgröße nicht durch Anzahl Prozesse pro Dimension teilbar!\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    int block_size = MATRIX_SIZE / q;
    int dims[2] = {q, q};
    int periods[2] = {1, 1};  // zyklische Anordnung
    int reorder = 0;
    MPI_Comm KART_KOMM;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &KART_KOMM);
    MPI_Comm_rank(KART_KOMM, &rank);
    MPI_Comm_size(KART_KOMM, &size);



    //Datatype für die Submatrix erstellen, einfachere Handhabung
    MPI_Datatype TYP_SUBMATRIX;
    MPI_Type_vector(block_size, block_size, MATRIX_SIZE, MPI_DOUBLE, &TYP_SUBMATRIX);
    MPI_Type_create_resized(TYP_SUBMATRIX, 0, block_size * sizeof(TYP), &TYP_SUBMATRIX);
    MPI_Type_commit(&TYP_SUBMATRIX);



    int koord[2];
    MPI_Cart_coords(KART_KOMM, rank, 2, koord);
    int my_rank_row = koord[0];
    int my_rank_col = koord[1];
    

    // Lokale Matrizen statt einzelner Elemente
    TYP* A_local = malloc(block_size * block_size * sizeof(TYP));
    TYP* B_local = malloc(block_size * block_size * sizeof(TYP));
    TYP* C_local = calloc(block_size * block_size, sizeof(TYP));



    TYP *matrix_a = malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(TYP));
    TYP *matrix_b = malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(TYP));
    

    int *sendcounts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));
    for (int i = 0; i < q; i++) {
        for (int j = 0; j < q; j++) {
            sendcounts[i * q + j] = 1;
            displs[i * q + j] = i * MATRIX_SIZE * block_size + j * block_size;
        }
    }

    if (rank == 0) {
        TYP *matrix_c = malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(TYP));
        befullenMatrix(matrix_a);
        befullenMatrix(matrix_b);
        multiplizierenMatrix(matrix_a, matrix_b, matrix_c);
/*         printf("Ergebniss Matrix (Korrekt):\n");
        ausgabeMatrix(matrix_c); */
        printf("Ergebniss Korrekt:"FORMAT"\n", berechenSummeArray(matrix_c));
        free(matrix_c);
    }

    MPI_Scatterv(matrix_a, sendcounts, displs, TYP_SUBMATRIX,
        A_local, block_size * block_size, IST_INT ? MPI_INT : MPI_DOUBLE,
        0, KART_KOMM);

    MPI_Scatterv(matrix_b, sendcounts, displs, TYP_SUBMATRIX,
            B_local, block_size * block_size, IST_INT ? MPI_INT : MPI_DOUBLE,
            0, KART_KOMM);

    // A initial BLÖCKE verschieben
    for (int k = 0; k < my_rank_row; k++) {
        int quell, ziel;
        MPI_Cart_shift(KART_KOMM, 1, -1, &quell, &ziel); // horizontaler shift (dimension 1)
        MPI_Sendrecv_replace(A_local, block_size*block_size, IST_INT ? MPI_INT : MPI_DOUBLE, ziel, 0, quell, 0, KART_KOMM, MPI_STATUS_IGNORE);
    }

    // B initial BLÖCKE verschieben
    for (int k = 0; k < my_rank_col; k++) {
        int quell, ziel;
        MPI_Cart_shift(KART_KOMM, 0, -1, &quell, &ziel); // vertikaler shift (dimension 0)
        MPI_Sendrecv_replace(B_local, block_size*block_size, IST_INT ? MPI_INT : MPI_DOUBLE, ziel, 1, quell, 1, KART_KOMM, MPI_STATUS_IGNORE);
    }
        
    
    for (int schritt = 0; schritt < q; schritt++) {
        // Lokale MATRIX Multiplikation
        for (int i = 0; i < block_size; i++) {
            for (int j = 0; j < block_size; j++) {
            for (int k = 0; k < block_size; k++) {
                C_local[i * block_size + j] += A_local[i * block_size + k] * B_local[k * block_size + j];
            }
            }
        }

    
        // Shift A nach links (horizontal)
        int ziel_A, quell_A;
        MPI_Cart_shift(KART_KOMM, 1, -1, &quell_A, &ziel_A);
        MPI_Sendrecv_replace(A_local, block_size*block_size, IST_INT ? MPI_INT : MPI_DOUBLE, ziel_A, 0, quell_A, 0, KART_KOMM, MPI_STATUS_IGNORE);
    
        // Shift B nach oben (vertikal)
        int ziel_B, quell_B;
        MPI_Cart_shift(KART_KOMM, 0, -1, &quell_B, &ziel_B);
        MPI_Sendrecv_replace(B_local, block_size*block_size, IST_INT ? MPI_INT : MPI_DOUBLE, ziel_B, 1, quell_B, 1, KART_KOMM, MPI_STATUS_IGNORE);
    }
       

    // End of Cannon Algorithmus: Sammeln der Ergebnisse
    TYP *C_result = NULL;
    if (rank == 0) {
        C_result = malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(TYP));
    }

    MPI_Gatherv(C_local, block_size * block_size, IST_INT ? MPI_INT : MPI_DOUBLE,
                C_result, sendcounts, displs, TYP_SUBMATRIX, 0, KART_KOMM);

    free(sendcounts);
    free(displs);

    // Ausgabe Ergebnis auf Rank 0
    if (rank == 0) {
/*         printf("Ergebnis Matrix:\n");
        ausgabeArrayAlsMatrix(C_result); */
        printf("Ergebnis (Cannon):" FORMAT "\n", berechenSummeArray(C_result));
        free(C_result);
    }

    free(A_local);
    free(B_local);
    free(C_local);
    free(matrix_a);
    free(matrix_b);
    MPI_Type_free(&TYP_SUBMATRIX);
    MPI_Comm_free(&KART_KOMM);
    MPI_Barrier(MPI_COMM_WORLD);

    ierr = MPI_Finalize();
    return 0;

}