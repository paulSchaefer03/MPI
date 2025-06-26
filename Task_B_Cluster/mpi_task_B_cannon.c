#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include "mpi.h"

#define MATRIX_SIZE 800
#define DETERMINISTIC 1 // 1 für deterministische Ausführung, 0 für zufällige Ausführung
#define CLUSTER 1       // 1 für Cluster, 0 für lokale Ausführung

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
    int rank, size, ierr;

    MPI_Status status;
    MPI_Init(&argc, &argv);

    MPI_Datatype MPI_TYP = IST_INT ? MPI_INT : MPI_DOUBLE;

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
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
    MPI_Comm_rank(KART_KOMM, &rank);
    MPI_Comm_size(KART_KOMM, &size);

    int koords[2];
    MPI_Cart_coords(KART_KOMM, rank, 2, koords);

    // Initialisierung der globalen Matrizen NUR auf Rank 0
    TYP *matrix_a = NULL, *matrix_b = NULL, *matrix_c = NULL;
    
    //Global INIT
    if (rank == 0) {
        //Speicher für die globalen Matrizen allokieren
        matrix_a = malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(TYP));
        matrix_b = malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(TYP));
        matrix_c = malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(TYP));
        // Initalisierung der globalen Matrizen

        befullenMatrix(matrix_a, DETERMINISTIC);
        //printf("A_Global:\n");
        //ausgabeArrayAlsMatrix(matrix_a, MATRIX_SIZE);
        
        befullenMatrix(matrix_b, DETERMINISTIC);
        //printf("B_Global:\n");
        //ausgabeArrayAlsMatrix(matrix_b, MATRIX_SIZE);


        // Check um später Richtigkeit zu prüfen
        if(MATRIX_SIZE <= 2000) {
            if(!CLUSTER){
                multiplizierenMatrix(matrix_a, matrix_b, matrix_c);

                //printf("Ergebniss Matrix (Korrekt):\n");
                //ausgabeArrayAlsMatrix(matrix_c, MATRIX_SIZE); 
                printf("Ergebniss Korrekt:"SUMTYPE_FORMAT"\n", berechenSummeArray(matrix_c));
            }

        }


    }

    double start_time = MPI_Wtime();

    // Broadcast der Submatrizen vorbereiten

    // Lokale Matrizen "Buffer" global für alle Ranks
    TYP* lokal_a = malloc(block_size * block_size * sizeof(TYP));
    TYP* lokal_b = malloc(block_size * block_size * sizeof(TYP));
    TYP *lokal_c = calloc(block_size * block_size, sizeof(TYP));

    int *sendzaehler = malloc(size * sizeof(int));
    int *verschiebungen = malloc(size * sizeof(int));

    // Index verschiebenungen auf den Matrizen berechenen um bei 1D Arrays und bei Scatterv die richtigen Daten zu senden
    for (int i = 0; i < size; i++) {
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

    // Scatterv für matrix_a
    MPI_Scatterv(matrix_a, sendzaehler, verschiebungen, TYPE_SUBMATRIX, lokal_a, block_size * block_size, MPI_TYP, 0, KART_KOMM);

    // Scatterv für matrix_b
    MPI_Scatterv(matrix_b, sendzaehler, verschiebungen, TYPE_SUBMATRIX, lokal_b, block_size * block_size, MPI_TYP, 0, KART_KOMM);



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

    if (rank == 0) {
        MPI_Gatherv(lokal_c, block_size * block_size, MPI_TYP, matrix_c, sendzaehler, verschiebungen, TYPE_SUBMATRIX, 0, KART_KOMM);
    } else {
        MPI_Gatherv(lokal_c, block_size * block_size, MPI_TYP, NULL, NULL, NULL, TYPE_SUBMATRIX, 0, KART_KOMM);
    }

    MPI_Barrier(KART_KOMM);
    double end_time = MPI_Wtime();
    if (rank == 0) {
        //printf("C_Global:\n");
        //ausgabeArrayAlsMatrix(matrix_c, MATRIX_SIZE);
        printf("Rechenzeit für Cannon-Algo(ohne Init aber mit Scatterv, Gatherv): %f Sekunden\n", end_time - start_time);
        schreibeMatrixInDatei("erg.txt", matrix_c);
        printf("Ergebniss Cannon:"SUMTYPE_FORMAT"\n", berechenSummeArray(matrix_c));
        
    }

    MPI_Barrier(KART_KOMM);
    
    //Aufräumen
/*     free(lokal_a); free(lokal_b); free(lokal_c);
    free(sendzaehler);free(verschiebungen);
    if (rank == 0) free(matrix_a);free(matrix_b);free(matrix_c);
    MPI_Type_free(&TYPE_SUBMATRIX); 
    MPI_Comm_free(&KART_KOMM);
    MPI_Barrier(MPI_COMM_WORLD); */

    ierr = MPI_Finalize();
    return 0;

}