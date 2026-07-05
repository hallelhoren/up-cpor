#include "ff.h"
#include "search.h"
#include <stdlib.h>

void ff_clear_hash_table(void) {
    int i;
    EhcHashEntry_pointer ehc_current, ehc_next;
    
    // Clear EHC Table
    for (i = 0; i < EHC_HASH_SIZE; i++) {
        ehc_current = gehc_hash_table[i];
        while (ehc_current != NULL) {
            ehc_next = ehc_current->next;
            if (ehc_current->S.F != NULL) { 
                 free(ehc_current->S.F);
            }
            free(ehc_current);
            ehc_current = ehc_next;
        }
        gehc_hash_table[i] = NULL;
    }
    
    BfsHashEntry_pointer bfs_current, bfs_next;
    
    // Clear BFS Table
    for (i = 0; i < BFS_HASH_SIZE; i++) {
        bfs_current = gbfs_hash_table[i];
        while (bfs_current != NULL) {
            bfs_next = bfs_current->next;
            if (bfs_current->S.F != NULL) {
                free(bfs_current->S.F);
            }
            free(bfs_current);
            bfs_current = bfs_next;
        }
        gbfs_hash_table[i] = NULL;
    }
}