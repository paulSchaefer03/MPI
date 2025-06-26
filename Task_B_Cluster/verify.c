#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MATRIX_SIZE 800  // Anpassbar auf deine Matrixgröße
#define TOLERANZ 1e-6    // Toleranz nur relevant bei double

// Hier wahlweise auf int oder double umstellen:
#define IST_INT 1
#if IST_INT
#define TYP int
#else
#define TYP double
#endif

void compare_files(const char* file1, const char* file2) {
    FILE *fp1 = fopen(file1, "r");
    FILE *fp2 = fopen(file2, "r");

    if (!fp1 || !fp2) {
        perror("Fehler beim Öffnen der Dateien");
        exit(EXIT_FAILURE);
    }

    char line1[10240], line2[10240];
    int row = 0;
    int differences = 0;

    while (fgets(line1, sizeof(line1), fp1) && fgets(line2, sizeof(line2), fp2)) {
        char *token1 = strtok(line1, ",\n");
        char *token2 = strtok(line2, ",\n");
        int col = 0;
        while (token1 && token2) {
            TYP val1, val2;
#if IST_INT
            val1 = atoi(token1);
            val2 = atoi(token2);
            if (val1 != val2) {
                printf("Unterschied bei Position [%d][%d]: Datei1=%d Datei2=%d\n", row, col, val1, val2);
                differences++;
                break;
            } else {
                printf("Position [%d][%d] ist identisch: %d\n", row, col, val1); // Optional: Ausgabe bei Identität
            }
#else
            val1 = atof(token1);
            val2 = atof(token2);
            if (fabs(val1 - val2) > TOLERANZ) {
                printf("Unterschied bei Position [%d][%d]: Datei1=%.10f Datei2=%.10f\n", row, col, val1, val2);
                differences++;
            }
#endif
            token1 = strtok(NULL, ",\n");
            token2 = strtok(NULL, ",\n");
            col++;
        }

        if (token1 || token2) {
            printf("Zeilenlaengen stimmen nicht in Zeile %d!\n", row);
            differences++;
        }
        row++;
    }

    if (fgets(line1, sizeof(line1), fp1) || fgets(line2, sizeof(line2), fp2)) {
        printf("Dateien haben unterschiedliche Anzahl an Zeilen!\n");
        differences++;
    }

    fclose(fp1);
    fclose(fp2);

    if (differences == 0) {
        printf("Die Dateien sind identisch.\n");
    } else {
        printf("Insgesamt %d Unterschiede gefunden.\n", differences);
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Benutzung: %s datei1.txt datei2.txt\n", argv[0]);
        return 1;
    }
    compare_files(argv[1], argv[2]);
    return 0;
}