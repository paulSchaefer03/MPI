#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define TYP int
#define IST_INT 1

void befullenMatrix(TYP *matrix, int MATRIX_SIZE, int deterministic, int reset) {
    static int seed_initialized = 0;
    if (!seed_initialized || reset) {
        if(deterministic){
            srand(42);// Feste Seed für Reproduzierbarkeit
            printf("Deterministische Initialisierung der Matrix mit fester Seed 42\n");
        } 
        else {
            srand(time(0));
        }
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

void multiplizierenMatrix(TYP *matrix_a, TYP *matrix_b, TYP *matrix_c, int MATRIX_SIZE) {
    #pragma omp parallel for
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            TYP sum = 0;
            for (int k = 0; k < MATRIX_SIZE; k++) {
                sum += matrix_a[i * MATRIX_SIZE + k] * matrix_b[k * MATRIX_SIZE + j];
            }
            matrix_c[i * MATRIX_SIZE + j] = sum;
        }
    }
}

long long berechneSumme(TYP *matrix, int size) {
    long long sum = 0;
    int n = size * size;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i++) {
        sum += matrix[i];
    }
    return sum;
}

int main() {//795, 798, 800, 1995, 2000, 4000, 4005, 4009, 6000, 6004, 7995, 
    int sizes[] = {7999, 8000};
    int anzahl = sizeof(sizes) / sizeof(sizes[0]);

    for (int i = 0; i < anzahl; i++) {
        int reset = 1;
        int size = sizes[i];
        printf("Matrixgröße: %d x %d\n", size, size);

        TYP *matrix_a = malloc(size * size * sizeof(TYP));
        TYP *matrix_b = malloc(size * size * sizeof(TYP));
        TYP *matrix_c = malloc(size * size * sizeof(TYP));

        befullenMatrix(matrix_a, size, 1, reset);  // deterministic = 1
        befullenMatrix(matrix_b, size, 1, !reset);  // deterministic = 1

        double start = omp_get_wtime();
        multiplizierenMatrix(matrix_a, matrix_b, matrix_c, size);
        double ende = omp_get_wtime();
        printf("  Multiplikationszeit: %.3f Sekunden\n", ende - start);

        long long sum = berechneSumme(matrix_c, size);
        printf("  Checksumme A: %lld\n\n", sum);

        free(matrix_a);
        free(matrix_b);
        free(matrix_c);
    }

    return 0;
}
