#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "mpi.h"

#define MATRIX_SIZE 16

// Formate für die Matrix-Elemente
#define TYP double
#define FORMAT "%f"
#define IST_INT 0
/* #define TYP int
#define FORMAT "%d" 
#define IST_INT 1 */

void ausgabeMatrix(TYP** arr) {
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            printf(FORMAT " ", arr[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

void ausgabeArrayAlsMatrix(TYP* arr) {
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            printf(FORMAT " ", arr[i * MATRIX_SIZE + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void befullenMatrix(TYP **matrix) {
    static int seed_initialized = 0;
    if (!seed_initialized) {
        srand(time(0));
        seed_initialized = 1;
    }

    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
#if IST_INT
            matrix[i][j] = (rand() % 20000001) - 10000000;
#else
            matrix[i][j] = ((rand() % 20000001) - 10000000) + ((double)rand() / RAND_MAX) * (rand() % 2 == 0 ? 1 : -1);
#endif
        }
    }

    //printf("Zeit zur Initialisierung der Matrix: %f Sekunden\n", ende - start);
}

void multiplizierenMatrix(TYP **matrix_a, TYP **matrix_b, TYP **matrix_c) {
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            matrix_c[i][j] = 0;
            for (int k = 0; k < MATRIX_SIZE; k++) {
                matrix_c[i][j] += matrix_a[i][k] * matrix_b[k][j];
            }
        }
    }
}

TYP berechneSummeMatrix(TYP **matrix) {
    TYP sum = 0;
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            sum += matrix[i][j];
        }
    }
    return sum;
}

TYP berechenSummeArray(TYP *array) {
    TYP sum = 0;
    for (int i = 0; i < MATRIX_SIZE*MATRIX_SIZE; i++) {
        sum += array[i];
    }
    return sum;
}

int vergleichenMatrix(TYP **matrix_a, TYP **matrix_b) {
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            if (matrix_a[i][j] != matrix_b[i][j]) {
                printf("Matrix mismatch at [%d][%d]: " FORMAT " != " FORMAT "\n", i, j, matrix_a[i][j], matrix_b[i][j]);
                return 0; // Matrices nicht gleich
            }
        }
    }
    return 1; // Matrices gleich
}

int main(int argc, char *argv[]) {

    int rank, size, ierr;


    TYP **matrix_a = (TYP **)malloc(MATRIX_SIZE * sizeof(TYP *));
    TYP **matrix_b = (TYP **)malloc(MATRIX_SIZE * sizeof(TYP *));
    
    MPI_Status status;
    MPI_Init(&argc, &argv);
    int dims[2] = {MATRIX_SIZE, MATRIX_SIZE};
    int periods[2] = {1, 1};  // zyklische Anordnung
    int reorder = 0;
    MPI_Comm KART_KOMM;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &KART_KOMM);
    MPI_Comm_rank(KART_KOMM, &rank);
    MPI_Comm_size(KART_KOMM, &size);

    int koord[2];
    MPI_Cart_coords(KART_KOMM, rank, 2, koord);
    int my_rank_row = koord[0];
    int my_rank_col = koord[1];
    
    for (int i = 0; i < MATRIX_SIZE; i++) {
        matrix_a[i] = (TYP *)malloc(MATRIX_SIZE * sizeof(TYP));
        matrix_b[i] = (TYP *)malloc(MATRIX_SIZE * sizeof(TYP));
    }

    TYP A_local, B_local, C_local = 0;

    if (rank == 0) {
        TYP **matrix_c = (TYP **)malloc(MATRIX_SIZE * sizeof(TYP *));
        for (int i = 0; i < MATRIX_SIZE; i++) {
            matrix_c[i] = (TYP *)malloc(MATRIX_SIZE * sizeof(TYP));
        }
        befullenMatrix(matrix_a);
        befullenMatrix(matrix_b);
        multiplizierenMatrix(matrix_a, matrix_b, matrix_c);
/*         printf("Ergebniss Matrix (Korrekt):\n");
        ausgabeMatrix(matrix_c); */
        printf("Ergebniss Korrekt:"FORMAT"\n", berechneSummeMatrix(matrix_c));
        free(matrix_c);

        for (int i = 0; i < MATRIX_SIZE; i++) {
            for (int j = 0; j < MATRIX_SIZE; j++) {
                int dest_rank;
                int dest_koords[2] = {i, j};
                MPI_Cart_rank(KART_KOMM, dest_koords, &dest_rank);
                if (dest_rank == 0) {
                    A_local = matrix_a[0][0];
                    B_local = matrix_b[0][0];
                } else {
                    MPI_Send(&matrix_a[i][j], 1, IST_INT ? MPI_INT : MPI_DOUBLE, dest_rank, 0, KART_KOMM);
                    MPI_Send(&matrix_b[i][j], 1, IST_INT ? MPI_INT : MPI_DOUBLE, dest_rank, 1, KART_KOMM);
                }
            }
        }

    }

    if (rank != 0) {
        MPI_Recv(&A_local, 1, IST_INT ? MPI_INT : MPI_DOUBLE, 0, 0, KART_KOMM, MPI_STATUS_IGNORE);
        MPI_Recv(&B_local, 1, IST_INT ? MPI_INT : MPI_DOUBLE, 0, 1, KART_KOMM, MPI_STATUS_IGNORE);
    } 

    // A initial verschieben
    for (int k = 0; k < my_rank_row; k++) {
        int quell, ziel;
        MPI_Cart_shift(KART_KOMM, 1, -1, &quell, &ziel); // horizontaler shift (dimension 1)
        MPI_Sendrecv_replace(&A_local, 1, IST_INT ? MPI_INT : MPI_DOUBLE, ziel, 0, quell, 0, KART_KOMM, MPI_STATUS_IGNORE);
    }

    // B initial verschieben
    for (int k = 0; k < my_rank_col; k++) {
        int quell, ziel;
        MPI_Cart_shift(KART_KOMM, 0, -1, &quell, &ziel); // vertikaler shift (dimension 0)
        MPI_Sendrecv_replace(&B_local, 1, IST_INT ? MPI_INT : MPI_DOUBLE, ziel, 1, quell, 1, KART_KOMM, MPI_STATUS_IGNORE);
    }
        
    
    for (int schritt = 0; schritt < MATRIX_SIZE; schritt++) {
        // Lokale Multiplikation
        C_local += A_local * B_local;
    
        // Shift A nach links (horizontal)
        int ziel_A, quell_A;
        MPI_Cart_shift(KART_KOMM, 1, -1, &quell_A, &ziel_A);
        MPI_Sendrecv_replace(&A_local, 1, IST_INT ? MPI_INT : MPI_DOUBLE, ziel_A, 0, quell_A, 0, KART_KOMM, MPI_STATUS_IGNORE);
    
        // Shift B nach oben (vertikal)
        int ziel_B, quell_B;
        MPI_Cart_shift(KART_KOMM, 0, -1, &quell_B, &ziel_B);
        MPI_Sendrecv_replace(&B_local, 1, IST_INT ? MPI_INT : MPI_DOUBLE, ziel_B, 1, quell_B, 1, KART_KOMM, MPI_STATUS_IGNORE);
    }
       

    // End of Cannon Algorithmus: Sammeln der Ergebnisse
    TYP *C_result = NULL;
    if (rank == 0) {
        C_result = malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(TYP));
    }

    MPI_Gather(&C_local, 1, IST_INT ? MPI_INT : MPI_DOUBLE, C_result, 1, IST_INT ? MPI_INT : MPI_DOUBLE, 0, KART_KOMM);

    // Ausgabe Ergebnis auf Rank 0
    if (rank == 0) {
/*         printf("Ergebnis Matrix:\n");
        ausgabeArrayAlsMatrix(C_result); */
        printf("Ergebnis (Cannon):"FORMAT"\n", berechenSummeArray(C_result));
    }

    
    ierr = MPI_Finalize();
    return 0;

}