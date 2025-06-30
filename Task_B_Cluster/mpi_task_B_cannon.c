#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include "mpi.h"

#define MATRIX_SIZE 2000
#define DETERMINISTIC 1 // 1 für deterministische Ausführung, 0 für zufällige Ausführung
#define CLUSTER 1       // 1 für Cluster, 0 für lokale Ausführung
#define VERTEILTES_INIT 0 // 1 für parallele Initialisierung durch alle Prozesse, 0 für Rank-0-only Init
#define CHECKSUMMEN 0

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
    
    int block_size = MATRIX_SIZE / q;


    int dims[2] = {q, q}, periods[2] = {1, 1};  // zyklische verschiebungen durch das gitter
    MPI_Comm KART_KOMM;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &KART_KOMM);
    // Bestimme den Rang des Prozesses im neuen Kommunikator (sollte sich nicht änderen da Kommunikatoren gleich groß bin mir aber nicht sicher)
    MPI_Comm_rank(KART_KOMM, &kart_rank);
    MPI_Comm_size(KART_KOMM, &kart_size);

    int koords[2];
    MPI_Cart_coords(KART_KOMM, kart_rank, 2, koords);

    // Initialisierung der globalen Matrizen NUR auf Rank 0
    TYP *matrix_a = NULL, *matrix_b = NULL;
    TYP *matrix_c = malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(TYP));
    
    if(kart_rank == 0) {
        printf("Matrix groesse : %d x %d\n", MATRIX_SIZE, MATRIX_SIZE);
        printf("Anzahl Ranks: %d\n", kart_size);
        printf("Blockgroesse: %d x %d\n", block_size, block_size);
        printf("Kommunikator: %d x %d\n", dims[0], dims[1]);
    }

    //Global INIT
#if VERTEILTES_INIT == 0
    if (kart_rank == 0) {
        //Speicher für die globalen Matrizen allokieren
        matrix_a = malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(TYP));
        matrix_b = malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(TYP));
        //matrix_c = malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(TYP));
        // Initalisierung der globalen Matrizen

        befullenMatrix(matrix_a, DETERMINISTIC);
        befullenMatrix(matrix_b, DETERMINISTIC);

        // Check um später Richtigkeit zu prüfen
        if(MATRIX_SIZE <= 8000) {
            if(!CLUSTER && CHECKSUMMEN){
                multiplizierenMatrix(matrix_a, matrix_b, matrix_c);
                printf("Ergebniss Korrekt:"SUMTYPE_FORMAT"\n", berechenSummeArray(matrix_c));
            }

        }


    }
    MPI_Barrier(KART_KOMM);
    double end_init_time = MPI_Wtime();
#endif

    // Broadcast der Submatrizen vorbereiten

    // Lokale Matrizen "Buffer" global für alle Ranks
    TYP *lokal_a = calloc(block_size * block_size, sizeof(TYP));
    TYP *lokal_b = calloc(block_size * block_size, sizeof(TYP));
    TYP *lokal_c = calloc(block_size * block_size, sizeof(TYP));

    int *sendzaehler = malloc(kart_size * sizeof(int));
    int *verschiebungen = malloc(kart_size * sizeof(int));

    // Index verschiebenungen auf den Matrizen berechenen um bei 1D Arrays und bei Scatterv die richtigen Daten zu senden
    for (int i = 0; i < kart_size; i++) {
        sendzaehler[i] = 1;
        int zeile = i / q;
        int spalte = i % q;
        verschiebungen[i] = zeile * MATRIX_SIZE * block_size + spalte * block_size;
    }


    //Datatypen für die Submatrizen erstellen, einfachere Handhabung
    MPI_Datatype TYPE_SUBMATRIX;
    MPI_Type_vector(block_size, block_size, MATRIX_SIZE, MPI_TYP, &TYPE_SUBMATRIX);
    MPI_Type_create_resized(TYPE_SUBMATRIX, 0, sizeof(TYP), &TYPE_SUBMATRIX);
    MPI_Type_commit(&TYPE_SUBMATRIX);

#if VERTEILTES_INIT == 0
    double start_scatter_time = MPI_Wtime();
    // Scatterv für matrix_a
    MPI_Scatterv(matrix_a, sendzaehler, verschiebungen, TYPE_SUBMATRIX, lokal_a, block_size * block_size, MPI_TYP, 0, KART_KOMM);

    // Scatterv für matrix_b
    MPI_Scatterv(matrix_b, sendzaehler, verschiebungen, TYPE_SUBMATRIX, lokal_b, block_size * block_size, MPI_TYP, 0, KART_KOMM);
    double end_scatter_time = MPI_Wtime();

#else
    double start_init_time = MPI_Wtime();
    int global_row = koords[0] * block_size;
    int global_col = koords[1] * block_size;
    befullenMatrixBlock(lokal_a, block_size, global_row, global_col, DETERMINISTIC);
    befullenMatrixBlock(lokal_b, block_size, global_row, global_col, DETERMINISTIC);
    double end_init_time = MPI_Wtime();
    if (kart_rank == 0) {
        printf("Initialisierung der lokalen Matrizen auf allen Ranks: %f Sekunden\n", end_init_time - start_init_time);
    }
#endif

    double start_cannon_time = MPI_Wtime();
    // Initiales Verschieben A: shift left (dimension 1), Blockweise nach links NICHT ELEMENTWEISE
    int quelle_A, ziel_A;
    MPI_Cart_shift(KART_KOMM, 1, -koords[0], &quelle_A, &ziel_A);
    MPI_Sendrecv_replace(lokal_a, block_size * block_size, MPI_TYP,
                        ziel_A, 0, quelle_A, 0, KART_KOMM, MPI_STATUS_IGNORE);
    
    
    // Initiales Verschieben B: shift up (dimension 0), Blockweise nach oben NICHT ELEMENTWEISE
    int quelle_B, ziel_B;
    MPI_Cart_shift(KART_KOMM, 0, -koords[1], &quelle_B, &ziel_B);
    MPI_Sendrecv_replace(lokal_b, block_size * block_size, MPI_TYP, ziel_B, 0, quelle_B, 0, KART_KOMM, MPI_STATUS_IGNORE);



    for (int schritt = 0; schritt < q; schritt++) {
        // Lokale Multiplikation
        for (int i = 0; i < block_size; i++) {
            for (int j = 0; j < block_size; j++) {
                for (int k = 0; k < block_size; k++) {
                    lokal_c[i * block_size + j] += lokal_a[i * block_size + k] * lokal_b[k * block_size + j];
                }
            }
        }

        // Danach: shiften
        if (schritt < q - 1) {  // Letzter Shift nicht mehr nötig nach q Schritten
            int quelle_A, ziel_A;
            MPI_Cart_shift(KART_KOMM, 1, -1, &quelle_A, &ziel_A);
            MPI_Sendrecv_replace(lokal_a, block_size * block_size, MPI_TYP, ziel_A, 0, quelle_A, 0, KART_KOMM, MPI_STATUS_IGNORE);

            int quelle_B, ziel_B;
            MPI_Cart_shift(KART_KOMM, 0, -1, &quelle_B, &ziel_B);
            MPI_Sendrecv_replace(lokal_b, block_size * block_size, MPI_TYP, ziel_B, 0, quelle_B, 0, KART_KOMM, MPI_STATUS_IGNORE);
            
        }
    }
    double end_cannon_time = MPI_Wtime();



    // Ausschließlich auf dem Cluster (NICHT LOKAL) kommt es mit Gatherv zu Problemen
    // Diese treten bei ganz bestimmten Matrizen größen mit bestimmter Anzahl Ranks auf
    // z.B. 800x800 mit 100 Ranks
    // Im Fehlerfall wird eine Matrix erzeug, in welcher mache Werte in bestimmten Zeilen um eine Zahl versetzt sind
    // Alle Submatrizen werden korrekt berechnet und der Fehler ist für mich nicht weiter zu debuggen
    // Problem ist ein Fehler in MPI nicht im Code, wie in der Übung geklärt
    double start_gather_time = MPI_Wtime();
    if (kart_rank == 0) {
        MPI_Gatherv(lokal_c, block_size * block_size, MPI_TYP, matrix_c, sendzaehler, verschiebungen, TYPE_SUBMATRIX, 0, KART_KOMM);
    } else {
        MPI_Gatherv(lokal_c, block_size * block_size, MPI_TYP, NULL, NULL, NULL, TYPE_SUBMATRIX, 0, KART_KOMM);
    } 


    //Alternative mit MPI_Gather
/*     if (kart_rank == 0) {
        // Empfangspuffer allokieren
        TYP *recv_buffer = malloc(kart_size * block_size * block_size * sizeof(TYP));
        
        // Alle Prozessoren senden lokal_c recv_buffer auf Rank 0
        MPI_Gather(lokal_c, block_size * block_size, MPI_TYP,
                recv_buffer, block_size * block_size, MPI_TYP,
                0, KART_KOMM);

        // Entpacken: Subblöcke einsortieren in matrix_c
        for (int p = 0; p < kart_size; ++p) {
            for (int i = 0; i < block_size; ++i) {
                int pi = p / q;
                int pj = p % q;

                // Quelladresse im Empfangspuffer
                TYP *quell = recv_buffer + p * block_size * block_size + i * block_size;

                // Zieladresse im globalen Ergebnisarray
                TYP *ziel = matrix_c + (pi * block_size + i) * MATRIX_SIZE + pj * block_size;

                memcpy(ziel, quell, block_size * sizeof(TYP));
            }
        }

        free(recv_buffer);
    } else {
        // Alle nicht-Rank-0 senden nur ihre Submatrix
        MPI_Gather(lokal_c, block_size * block_size, MPI_TYP, NULL, 0, MPI_TYP, 0, KART_KOMM);
    } */

    double end_gather_time = MPI_Wtime();

    MPI_Barrier(MPI_COMM_WORLD);
    double programm_end_time = MPI_Wtime();

    if (world_rank == 0) {
        double duration = programm_end_time - programm_start_time;
        printf("Initialisierung der Matrizen auf Rank 0: %f Sekunden\n", end_init_time - start_init_time);
        printf("Scatterv Zeit: %f Sekunden\n", end_scatter_time - start_scatter_time);
        printf("Cannon Rechenzeit: %f Sekunden\n", end_cannon_time - start_cannon_time);
        printf("Gatherv Zeit: %f Sekunden\n", end_gather_time - start_gather_time);
#if CHECKSUMMEN
        printf("Ergebnis Cannon: " SUMTYPE_FORMAT "\n", berechenSummeArray(matrix_c));
#endif
        printf("Programm ausgeführt in: %f Sekunden\n", duration);
    }

    // Speicher freigeben   
    if (world_rank == 0){
        free(matrix_a);free(matrix_b);free(matrix_c);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    free(lokal_a); free(lokal_b); free(lokal_c);
    free(sendzaehler);
    free(verschiebungen);
    MPI_Type_free(&TYPE_SUBMATRIX); 
    MPI_Comm_free(&KART_KOMM);
    MPI_Barrier(MPI_COMM_WORLD); 

    MPI_Finalize();

    return 0;

}