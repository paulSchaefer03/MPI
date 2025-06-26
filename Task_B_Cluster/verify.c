#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TOLERANZ 1e-6
#define IST_INT 1

#if IST_INT
    #define TYP int
#else
    #define TYP double
#endif

void compare_files(const char* file1, const char* file2, const char* outfile) {
    FILE *fp1 = fopen(file1, "r");
    FILE *fp2 = fopen(file2, "r");
    FILE *out = fopen(outfile, "w");
    if (!fp1 || !fp2 || !out) {
        perror("Fehler beim Öffnen der Dateien");
        exit(EXIT_FAILURE);
    }

    char *line1 = NULL, *line2 = NULL;
    size_t len1 = 0, len2 = 0;
    size_t read1, read2;
    int row = 0, differences = 0;

    while (1) {
        read1 = getline(&line1, &len1, fp1);
        read2 = getline(&line2, &len2, fp2);

        if (read1 == -1 && read2 == -1)
            break;

        // Ignoriere leere Zeilen am Ende
        if (read1 != -1 && strspn(line1, " \t\r\n") == strlen(line1)) continue;
        if (read2 != -1 && strspn(line2, " \t\r\n") == strlen(line2)) continue;

        if (read1 == -1 || read2 == -1) {
            fprintf(out, "Unterschiedliche Zeilenanzahl festgestellt!\n");
            differences++;
            break;
        }
        row++;

        // Entferne ggf. das '\n'
        if (line1[read1 - 1] == '\n') line1[--read1] = '\0';
        if (line2[read2 - 1] == '\n') line2[--read2] = '\0';

        char *p1 = line1;
        char *p2 = line2;
        int col = 0;

        while (*p1 && *p2) {
            char *end1, *end2;
            TYP val1, val2;

#if IST_INT
            val1 = (TYP)strtol(p1, &end1, 10);
            val2 = (TYP)strtol(p2, &end2, 10);
            if (val1 != val2) {
                fprintf(out, "Unterschied bei Position [%d][%d]: Datei1=%d Datei2=%d\n", row - 1, col, val1, val2);
                differences++;
            }
#else
            val1 = strtod(p1, &end1);
            val2 = strtod(p2, &end2);
            if (fabs(val1 - val2) > TOLERANZ) {
                fprintf(out, "Unterschied bei Position [%d][%d]: Datei1=%.10f Datei2=%.10f\n", row - 1, col, val1, val2);
                differences++;
            }
#endif
            if (p1 == end1 || p2 == end2) {
                fprintf(out, "Parsing-Fehler in Zeile %d Spalte %d!\n", row - 1, col);
                differences++;
                break;
            }

            p1 = (*end1 == ',') ? end1 + 1 : end1;
            p2 = (*end2 == ',') ? end2 + 1 : end2;
            col++;
        }

        if (*p1 || *p2) {
            fprintf(out, "Unterschiedliche Spaltenzahl in Zeile %d!\n", row - 1);
            differences++;
        }
    }

    if ((read1 != -1 && read2 == -1) || (read1 == -1 && read2 != -1)) {
        fprintf(out, "Unterschiedliche Zeilenanzahl festgestellt!\n");
        differences++;
    }

    free(line1);
    free(line2);
    fclose(fp1);
    fclose(fp2);

    if (differences == 0) {
        fprintf(out, "Die Dateien sind identisch.\n");
        printf("Die Dateien sind identisch.\n");
    } else {
        fprintf(out, "Insgesamt %d Unterschiede gefunden.\n", differences);
        printf("Insgesamt %d Unterschiede gefunden. Siehe %s\n", differences, outfile);
    }
    fclose(out);
}

int main(int argc, char *argv[]) {
    if (argc == 4) {
        compare_files(argv[1], argv[2], argv[3]);
    } else if (argc == 2 && strcmp(argv[1], "--batch") == 0) {
        FILE *out = fopen("verify_results.txt", "w");
        if (!out) {
            perror("Fehler beim Öffnen von verify_results.txt");
            return 1;
        }
        int total_differences = 0;
        for (int i = 0; i < 100; ++i) {
            char file1[128], file2[128];
            snprintf(file1, sizeof(file1), "submatrix_local_rank_%d.txt", i);
            snprintf(file2, sizeof(file2), "submatrix_cluster_rank_%d.txt", i);

            // Temporäre Datei für Zwischenergebnis
            char tmpfile[] = "verify_tmp.txt";
            compare_files(file1, file2, tmpfile);

            // Schreibe Ergebnis in verify_results.txt
            FILE *tmp = fopen(tmpfile, "r");
            if (tmp) {
                fprintf(out, "Vergleich %s <-> %s:\n", file1, file2);
                char buf[512];
                while (fgets(buf, sizeof(buf), tmp)) {
                    fputs(buf, out);
                    if (strstr(buf, "Unterschied") && !strstr(buf, "identisch")) {
                        total_differences++;
                    }
                }
                fprintf(out, "\n");
                fclose(tmp);
                remove(tmpfile);
            } else {
                fprintf(out, "Fehler beim Öffnen von %s\n\n", tmpfile);
            }
        }
        fprintf(out, "Batch-Vergleich abgeschlossen. Unterschiede in %d von 100 Vergleichen.\n", total_differences);
        fclose(out);
        printf("Batch-Vergleich abgeschlossen. Ergebnisse in verify_results.txt\n");
    } else {
        fprintf(stderr, "Benutzung:\n");
        fprintf(stderr, "  %s datei1.txt datei2.txt unterschiede.txt\n", argv[0]);
        fprintf(stderr, "  %s --batch\n", argv[0]);
        return 1;
    }
    return 0;
}
