#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include "mpi.h"

#define MATRIX_SIZE 2000
#define DETERMINISTIC 1 // 1 für deterministische Ausführung(verifikation der Checksummen), 0 für zufällige Ausführung
#define CLUSTER 1       // 1 für Cluster, 0 für lokale Ausführung
#define CHECKSUMMEN 1   // 1 für Checksummen, 0 für keine Checksummen

// Formate für die Matrix-Elemente

/* #define TYP double
#define FORMAT "%f"
#define SUMTYPE_FORMAT "%Lf"
#define IST_INT 0 
#define SUMTYPE long double */

#define TYP int
#define FORMAT "%d" 
#define SUMTYPE_FORMAT "%lld"
#define IST_INT 1
#define SUMTYPE long long


#if CLUSTER
#define FILE_NAME "erg_cluster.txt"
#define FILE_NAME_MATRIX_A "matrix_a_cluster.txt"
#define FILE_NAME_MATRIX_B "matrix_b_cluster.txt"
#else
#define FILE_NAME "erg_local.txt"
#define FILE_NAME_MATRIX_A "matrix_a_local.txt"
#define FILE_NAME_MATRIX_B "matrix_b_local.txt"
#endif



void schreibeMatrixInDatei(const char *dateiname, TYP *matrix) {
    FILE *fp = fopen(dateiname, "w");
    if (fp == NULL) {
        perror("Fehler beim Öffnen der Datei");
        return;
    }

    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            fprintf(fp, FORMAT, matrix[i * MATRIX_SIZE + j]);
            if (j < MATRIX_SIZE - 1)
                fprintf(fp, ",");
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
}


void ausgabeArrayAlsMatrix(TYP* arr, int block_size) {
    for (int i = 0; i < block_size; i++) {
        for (int j = 0; j < block_size; j++) {
            printf(FORMAT " ", arr[i * block_size + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void befullenMatrix(TYP *matrix, int deterministic) {
    static int seed_initialized = 0;
    if (!seed_initialized) {
        if(deterministic) srand(42);// Feste Seed für Reproduzierbarkeit
        else srand(time(0));
        seed_initialized = 1;
    }

    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
#if IST_INT
        matrix[i] = (rand() % 200001) - 100000;
#else
        matrix[i] = ((rand() % 200001) - 100000) + ((double)rand() / RAND_MAX) * (rand() % 2 == 0 ? 1 : -1);
#endif
    }
}

void befullenMatrixBlock(TYP *matrix, int block_size, int global_row_offset, int global_col_offset, int deterministic) {
    unsigned int seed = deterministic ? 42 + global_row_offset * MATRIX_SIZE + global_col_offset : (unsigned int)time(NULL);
    for (int i = 0; i < block_size; i++) {
        for (int j = 0; j < block_size; j++) {
            int idx = i * block_size + j;
#if IST_INT
            matrix[idx] = (rand_r(&seed) % 200001) - 100000;
#else
            matrix[idx] = ((rand_r(&seed) % 200001) - 100000) + ((double)rand_r(&seed) / RAND_MAX) * (rand_r(&seed) % 2 == 0 ? 1 : -1);
#endif
        }
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


SUMTYPE berechenSummeArray(TYP *array) {
    SUMTYPE sum = 0;
    for (int i = 0; i < MATRIX_SIZE*MATRIX_SIZE; i++) {
        sum += array[i];
    }
    return sum;
}


int main(int argc, char *argv[]) {
    
    int world_rank, world_size;
    int kart_rank, kart_size, ierr;

    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Datatype MPI_TYP = IST_INT ? MPI_INT : MPI_DOUBLE;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    MPI_Barrier(MPI_COMM_WORLD);
    double programm_start_time = MPI_Wtime();
    double start_init_time = MPI_Wtime();
    int q = sqrt(world_size);  // Prozessgittergröße

    if (q * q != world_size) {
        if (world_rank == 0) {
            printf("Anzahl der Prozesse muss eine Quadratzahl sein!\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (MATRIX_SIZE % q != 0) {
        if (world_rank == 0) {
            printf("Matrixgröße nicht durch Anzahl Prozesse pro Dimension teilbar!\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    //int block_size = MATRIX_SIZE / q;
    int block_size = MATRIX_SIZE / q;
    int block_blocksize = block_size * block_size;

    // Initialisierung der globalen Matrizen NUR auf Rank 0
    TYP *global_blocks_a = NULL, *global_blocks_b = NULL; TYP *global_blocks_c = NULL;
    TYP *lokal_a = malloc(block_blocksize * sizeof(TYP));
    TYP *lokal_b = malloc(block_blocksize * sizeof(TYP));
    TYP *lokal_c = calloc(block_blocksize, sizeof(TYP));

    TYP *tmp_a = malloc(block_blocksize * sizeof(TYP));
    TYP *tmp_b = malloc(block_blocksize * sizeof(TYP));


    int dims[2] = {q, q}, periods[2] = {1, 1};  // zyklische verschiebungen durch das gitter
    MPI_Comm KART_KOMM;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &KART_KOMM);
    // Bestimme den Rang des Prozesses im neuen Kommunikator (sollte sich nicht änderen da Kommunikatoren gleich groß bin mir aber nicht sicher)
    MPI_Comm_rank(KART_KOMM, &kart_rank);
    MPI_Comm_size(KART_KOMM, &kart_size);

    int koords[2];
    MPI_Cart_coords(KART_KOMM, kart_rank, 2, koords);



    if(kart_rank == 0) {
        printf("Matrix groesse : %d x %d\n", MATRIX_SIZE, MATRIX_SIZE);
        printf("Anzahl Ranks: %d\n", kart_size);
        printf("Blockgroesse: %d x %d\n", block_size, block_size);
        printf("Kommunikator: %d x %d\n", dims[0], dims[1]);
    }

    //Global INIT
    if (kart_rank == 0) {
        TYP *matrix_a_full = malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(TYP));
        TYP *matrix_b_full = malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(TYP));
        global_blocks_a = malloc(kart_size * block_blocksize * sizeof(TYP));
        global_blocks_b = malloc(kart_size * block_blocksize * sizeof(TYP));

        befullenMatrix(matrix_a_full, DETERMINISTIC);
        befullenMatrix(matrix_b_full, DETERMINISTIC);

        for (int bi = 0; bi < q; ++bi) {
            for (int bj = 0; bj < q; ++bj) {
                int block_index = bi * q + bj;
                TYP *ziel_a = global_blocks_a + block_index * block_blocksize;
                TYP *ziel_b = global_blocks_b + block_index * block_blocksize;
                for (int i = 0; i < block_size; ++i) {
                    memcpy(
                        ziel_a + i * block_size,
                        matrix_a_full + (bi * block_size + i) * MATRIX_SIZE + bj * block_size,
                        block_size * sizeof(TYP)
                    );
                    memcpy(
                        ziel_b + i * block_size,
                        matrix_b_full + (bi * block_size + i) * MATRIX_SIZE + bj * block_size,
                        block_size * sizeof(TYP)
                    );
                }
            }
        }
        // Check um später Richtigkeit zu prüfen
        if(MATRIX_SIZE <= 8000) {
            if(!CLUSTER && CHECKSUMMEN){
                TYP *check_matrix_c = malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(TYP));
                multiplizierenMatrix(matrix_a_full, matrix_b_full, check_matrix_c);
                //printf("Ergebniss Matrix (Korrekt):\n");
                //ausgabeArrayAlsMatrix(check_matrix_c, MATRIX_SIZE); 
                printf("Ergebniss Korrekt:"SUMTYPE_FORMAT"\n", berechenSummeArray(check_matrix_c));
                free(check_matrix_c);
            }

        }

        free(matrix_a_full);
        free(matrix_b_full);
    }
    

    MPI_Barrier(KART_KOMM);
    double end_init_time = MPI_Wtime();


    double start_scatter_time = MPI_Wtime();
    MPI_Scatter(global_blocks_a, block_blocksize, MPI_TYP, lokal_a, block_blocksize, MPI_TYP, 0, KART_KOMM);
    MPI_Scatter(global_blocks_b, block_blocksize, MPI_TYP, lokal_b, block_blocksize, MPI_TYP, 0, KART_KOMM);

    double end_scatter_time = MPI_Wtime();

    double start_cannon_time = MPI_Wtime();
    // Initiales Verschieben A: shift left (dimension 1), Blockweise nach links NICHT ELEMENTWEISE
    int quelle_A, ziel_A;
    MPI_Cart_shift(KART_KOMM, 1, -koords[0], &quelle_A, &ziel_A);
    MPI_Sendrecv(lokal_a, block_blocksize, MPI_TYP, 
                ziel_A, 0,
                tmp_a, block_blocksize, MPI_TYP,
                quelle_A, 0,
                KART_KOMM, MPI_STATUS_IGNORE);
    memcpy(lokal_a, tmp_a, block_blocksize * sizeof(TYP));

    // Initial-Shift B: up
    int quelle_B, ziel_B;
    MPI_Cart_shift(KART_KOMM, 0, -koords[1], &quelle_B, &ziel_B);
    MPI_Sendrecv(lokal_b, block_blocksize, MPI_TYP,
                ziel_B, 0,
                tmp_b, block_blocksize, MPI_TYP,
                quelle_B, 0,
                KART_KOMM, MPI_STATUS_IGNORE);
    memcpy(lokal_b, tmp_b, block_blocksize * sizeof(TYP));

    for (int schritt = 0; schritt < q; schritt++) {
        // Optimierte lokale Multiplikation
        for (int i = 0; i < block_size; ++i) {
            for (int k = 0; k < block_size; ++k) {
                TYP a_val = lokal_a[i * block_size + k];
                for (int j = 0; j < block_size; ++j) {
                    lokal_c[i * block_size + j] += a_val * lokal_b[k * block_size + j];
                }
            }
        }

        if (schritt < q - 1) {
            // Shift A left
            MPI_Cart_shift(KART_KOMM, 1, -1, &quelle_A, &ziel_A);
            MPI_Sendrecv(lokal_a, block_blocksize, MPI_TYP,
                        ziel_A, 0,
                        tmp_a, block_blocksize, MPI_TYP,
                        quelle_A, 0,
                        KART_KOMM, MPI_STATUS_IGNORE);
            memcpy(lokal_a, tmp_a, block_blocksize * sizeof(TYP));

            // Shift B up
            MPI_Cart_shift(KART_KOMM, 0, -1, &quelle_B, &ziel_B);
            MPI_Sendrecv(lokal_b, block_blocksize, MPI_TYP,
                        ziel_B, 0,
                        tmp_b, block_blocksize, MPI_TYP,
                        quelle_B, 0,
                        KART_KOMM, MPI_STATUS_IGNORE);
            memcpy(lokal_b, tmp_b, block_blocksize * sizeof(TYP));
        }
    }
    double end_cannon_time = MPI_Wtime();

    double start_gather_time = MPI_Wtime();

    if (kart_rank == 0) {
        global_blocks_c = malloc(kart_size * block_blocksize * sizeof(TYP));
    }

    MPI_Gather(lokal_c, block_blocksize, MPI_TYP, global_blocks_c, block_blocksize, MPI_TYP, 0, KART_KOMM);


    if (kart_rank == 0) {
        TYP *matrix_c = malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(TYP));
        for (int bi = 0; bi < q; ++bi) {
            for (int bj = 0; bj < q; ++bj) {
                int block_index = bi * q + bj;
                TYP *quelle = global_blocks_c + block_index * block_blocksize;
                for (int i = 0; i < block_size; ++i) {
                    memcpy(
                        matrix_c + (bi * block_size + i) * MATRIX_SIZE + bj * block_size,
                        quelle + i * block_size,
                        block_size * sizeof(TYP)
                    );
                }
            }
        }
        double end_gather_time = MPI_Wtime();
        double programm_end_time = MPI_Wtime();
        double duration = programm_end_time - programm_start_time;
        printf("Initialisierung der Matrizen auf Rank 0: %f Sekunden\n", end_init_time - start_init_time);
        printf("Scatter Zeit: %f Sekunden\n", end_scatter_time - start_scatter_time);
        printf("Cannon Rechenzeit: %f Sekunden\n", end_cannon_time - start_cannon_time);
        printf("Gather Zeit: %f Sekunden\n", end_gather_time - start_gather_time);
#if CHECKSUMMEN
        printf("Ergebnis Cannon: " SUMTYPE_FORMAT "\n", berechenSummeArray(matrix_c));
#endif
        printf("Programm ausgeführt in: %f Sekunden\n", duration);
        free(matrix_c);
    }



    MPI_Barrier(MPI_COMM_WORLD);
    
    // Speicher freigeben   
    free(lokal_a);
    free(lokal_b);
    free(lokal_c);
    if (kart_rank == 0) {
        free(global_blocks_a);
        free(global_blocks_b);
        free(global_blocks_c);
    }

    MPI_Comm_free(&KART_KOMM);
    MPI_Barrier(MPI_COMM_WORLD); 

    MPI_Finalize();

    return 0;

}