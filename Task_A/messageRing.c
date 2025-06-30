#include <stdio.h>
#include <unistd.h> // für sleep()
#include <mpi.h>

#define TAG_to_right 1

int main(int argc, char *argv[]) {

    int my_rank, my_size;
    int snd_buf, rcv_buf;
    int right, left;
    int sum, i;
    MPI_Status status;
    MPI_Request request;

    MPI_Init(&argc, &argv); 
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &my_size);
    
    right = (my_rank+1) % my_size; 
    left = (my_rank-1+my_size) % my_size;
    sum = 0;

    for(i = 0; i < my_size; i++) { 
        if(my_rank == i) {// Immer nur der aktuelle Rank
            if (my_rank == 0) {
                snd_buf = 0;
            } else {
                printf("I am Rank %d and I want to receive a message\n", my_rank);
                MPI_Recv(&rcv_buf, 1, MPI_INT, left, TAG_to_right, MPI_COMM_WORLD, &status);
                snd_buf = rcv_buf;
            }

            printf("I am Rank %d and I send Message %d\n", my_rank, snd_buf + 1);
            snd_buf++; // Inkrementieren
            sum = snd_buf;

            // Aufgabe C sleep vor dem Senden, 
            // jeder Rank will sofort empfangen das senden geht aber immer erst mit 1 sek delay
            // und wenn er die Nachricht vom linken Nachbarn empfangen hat
            //sleep(1);

            if (my_rank != my_size - 1) {
                MPI_Send(&snd_buf, 1, MPI_INT, right, TAG_to_right, MPI_COMM_WORLD);
            }
 
        }
        //MPI_Barrier(MPI_COMM_WORLD);
        // Aufgabe D sleep nach der Iteration,
        // der nächste Rank wartet 1 Sekunde bevor er die nächste Nachricht empfangen will, 
        // sendet aber sofort die Antwort 
        //sleep(1);
        
    }
    //Ohne sleep() wollen alle Ranks wollen empfangen aber nur der erste Rank sendet
    //die Ausgabe ist nicht deterministisch, da die Ranks nicht synchronisiert sind
    printf("Rank %i:\tSum = %i\n", my_rank, sum); 
    MPI_Finalize();
    return 0;
}
