#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <lapacke.h>
#include <stdlib.h>
#include <string.h>
// Compiler Flags: -O3 -fopenmp -llapacke -llapack -lblas -lm

#define N 3 // The Dimension of Matrices, no '=' sign
static double lower_bound = 1.001; // To avoid rounding errors, round slightly up from integers
static double upper_bound = 1.999; // To avoid rounding errors, round slightly down from integers
static int max_digit = 10;
int mode = 3; // Defines the Mode for displaying information

// Optional Switches
int skip_all_pruning = 0; // TURNS PRUNING OFF! FOR COUNTING ONLY!!! SET MODE = 0
int no_a_to_b_squared = 0; // No mapping to a doubled edge alone
int oniontime = 0; // Column Parity Must Match
int final_column_sums_to_three = 0; // Final Column Sums To 3
int n_plus_two = 0; // Entries sum to N + 2

// -------------------------------------------------------- //
static double wallStart = 0;
#define MAX_UNIQUE_EIGENVALUES 1000000
static double unique_eigenvalues[MAX_UNIQUE_EIGENVALUES];
static int unique_count = 0;
double var_to_not_exceed;
double eigen_tolerance = 1e-5;
// -------------------------------------------------------- //

/* ALIGNMENT FIX: Padding the struct to 64 bytes (typical L1 cache line size).
This was added by the AI to avoid distinct threads overriding each other's work. */
typedef struct {
    double* eigenvalues;
    long long visited;
    int     count;
    int     increment;
    int     eigenwaste;
    char    padding[64]; // Pad out the struct to prevent cache line collisions
} __attribute__((aligned(64))) ThreadLocal;

// -------------------------------------------------------- //

static int matrixSum(int a[N * N]) { // Sums the entries of a matrix
    int sum = 0;
    for (int i = 0; i < N * N; i++) {
        sum += a[i];
    }
    return sum;
}

static int checkColumnParity(int a[N * N]) { // Checks if columns have the same parity
    int first_col_sum = 0;
    for (int row = 0; row < N; row++) {
        first_col_sum += a[row * N + 0];
    }
    int reference_parity = first_col_sum % 2;

    for (int col = 1; col < N; col++) {
        int col_sum = 0;
        for (int row = 0; row < N; row++) {
            col_sum += a[row * N + col];
        }
        if (col_sum % 2 != reference_parity) {
            return 0; // Returns 0 if at least one column differs from the first column's parity
        }
    }
    return 1;
}

static int checkRowsHaveSameUniqueOne(int a[N * N]) {
    for (int r1 = 0; r1 < N; r1++) {
        for (int r2 = r1 + 1; r2 < N; r2++) {
            int col_idx1 = -1;
            int col_idx2 = -1;
            int count1 = 0, count2 = 0;

            for (int c = 0; c < N; c++) {
                if (a[r1 * N + c] != 0) {
                    count1++;
                    if (a[r1 * N + c] == 1) col_idx1 = c;
                }
            }
            for (int c = 0; c < N; c++) {
                if (a[r2 * N + c] != 0) {
                    count2++;
                    if (a[r2 * N + c] == 1) col_idx2 = c;
                }
            }

            // Condition: Both rows have exactly one 1 and no other non-zero values
            if (count1 == 1 && count2 == 1 && col_idx1 == col_idx2 && col_idx1 != -1) {
                return 1;
            }
        }
    }
    return 0; // No Lonely Towers
}

static void printMatrix(int a[N * N]) { // Formats how matrices are printed
    printf("[");
    for (int i = 0; i < N * N; i++)
        printf("%d%s", a[i], i < N * N - 1 ? ", " : "");
    printf("]");
}

int compare_doubles(const void* a, const void* b) { // For sorting eigenvalues to display later
    double da = *(const double*)a;
    double db = *(const double*)b;
    return (da > db) - (da < db);
}

double eig(double a_copy[N * N], double* work, int lwork) {
    int n = N; int lda = N; int info;
    double wr[N];
    double wi[N];

    info = LAPACKE_dgeev_work(LAPACK_ROW_MAJOR, 'N', 'N', n,
    a_copy, lda, wr, wi, NULL, n, NULL, n, work, lwork); // Calculates the matrix's eigenvalues

    if (info != 0) {
        return NAN;
    }

    double max_ev_sq = 0.0;
    for (int i = 0; i < n; i++) {
        double magnitude_sq = wr[i] * wr[i] + wi[i] * wi[i];
        if (magnitude_sq > max_ev_sq) {
            max_ev_sq = magnitude_sq;
        }
    }

    return sqrt(max_ev_sq); // Squares, then square roots the eigenvalues for the largest magnitude
}

static int irreducible(int a[N * N]) { // Checks if every edge can reach every other edge
    int visited_nodes[N] = {0};
    int stack[N], top = 0;
    stack[top++] = 0;
    visited_nodes[0] = 1;
    while (top > 0) {
        int u = stack[--top];
        for (int v = 0; v < N; v++)
            if (a[u * N + v] > 0 && !visited_nodes[v]) {
                visited_nodes[v] = 1;
                stack[top++] = v;
            }
    }
    for (int i = 0; i < N; i++)
        if (!visited_nodes[i]) return 0;

    int visited2[N] = {0};
    top = 0;
    stack[top++] = 0;
    visited2[0] = 1;
    while (top > 0) {
        int u = stack[--top];
        for (int v = 0; v < N; v++)
            if (a[v * N + u] > 0 && !visited2[v]) {
                visited2[v] = 1;
                stack[top++] = v;
            }
    }
    for (int i = 0; i < N; i++)
        if (!visited2[i]) return 0;

    return 1;
}

static void process(int a[N * N], double* work, int lwork, ThreadLocal* tl) {

    if (skip_all_pruning == 0) {if (!irreducible(a)) return;}

    if (skip_all_pruning == 0) { // Collatz-Wielandt Formula pruning, run over 8 rounds
        double x[N];
        double next_x[N]; // The new vectors from (I + a) * {x}, leaving \vec{x} untouched
        for (int i = 0; i < N; i++) x[i] = 1.0;

        int steps = 8;
        int satisfies_bounds = 1;

        for (int iter = 0; iter < steps; iter++) {
            for (int i = 0; i < N; i++) {
                double sum = x[i]; // Works the identity matrix into the computation
                for (int j = 0; j < N; j++) {
                    sum += a[i * N + j] * x[j]; // Computes (I + a) * {x} for positive products
                }
                next_x[i] = sum;
            }

            double min_q = next_x[0] / x[0];
            double max_q = next_x[0] / x[0];
            for (int i = 1; i < N; i++) {
                double q = next_x[i] / x[i];
                if (q < min_q) min_q = q; // Selects for the minimum component
                if (q > max_q) max_q = q; // Selects for the maximum component
            }

            double true_min = min_q - 1.0; // -1.0 because of adding the identity earlier
            double true_max = max_q - 1.0; // -1.0 because of adding the identity earlier

            if (true_max < lower_bound || true_min > upper_bound) {
                satisfies_bounds = 0; // In case the regular upper/lower bounds are violated
                break;
            }

            double norm = 0.0;
            for (int i = 0; i < N; i++) norm += next_x[i] * next_x[i];
            norm = sqrt(norm); // Gets the magnitude of the \vec{next_x}
            for (int i = 0; i < N; i++) x[i] = next_x[i] / norm; // Normalizes \vec{x}
        }

        if (!satisfies_bounds) {
            return;
        }
    }

    if (skip_all_pruning == 0) {
        if (checkRowsHaveSameUniqueOne(a)) return;
    }

    if (oniontime == 1 && skip_all_pruning == 0) {
        if (checkColumnParity(a) == 0) return;
    }

    if (final_column_sums_to_three == 1 && skip_all_pruning == 0) { // In case the col sums to 3
        if (matrixSum(a) != N + 2) return;
        int sharkplaceholder = 0;
        for (int i = 1; i <= N; i++) {
            sharkplaceholder += a[i * N - 1];
        } if (sharkplaceholder != 3) return;
    }

    double ad[N * N]; // To turn the integer array into a double array
    for (int i = 0; i < N * N; i++) {
        ad[i] = (double)a[i];
    }

    double ev = eig(ad, work, lwork); // Actually calculates the eigenvalue

    if (skip_all_pruning == 0) {
        if (ev < lower_bound || ev > upper_bound || isnan(ev)) {
            tl->eigenwaste++; // If the eigenvalue was calculated but out of bounds, + eigenwaste
            return;
        }
    }

    tl->increment++;

    if (mode == 1 || mode == 3) { // Settings to print eigenvalues
        #pragma omp critical
        {
            printf("%.10f with array ", ev);
            printMatrix(a);
            printf("\n");
            fflush(stdout);
        }
    }

    if (mode >= 2) { // Settings to display eigenvalues
        int found = 0;
        for (int i = 0; i < tl->count; i++) {
            if (fabs(tl->eigenvalues[i] - ev) < eigen_tolerance) {
                found = 1;
                break;
            }
        }
        if (!found && tl->count < MAX_UNIQUE_EIGENVALUES) {
            tl->eigenvalues[tl->count] = ev;
            tl->count++;
        }
    }
}

static void search(int *a, int pos, int max_digit, double* work, int lwork, ThreadLocal* tl, int tid) {
    if (pos == N * N) {
        tl->visited++;

        // Progress print isolated to thread 0
        if (tid == 0 && tl->visited % 100000000 == 0) {
             printf("Thread 0 reached %lld local matrices in %.5f seconds...\n",
                    tl->visited, omp_get_wtime() - wallStart);
             fflush(stdout);
        }

        process(a, work, lwork, tl);
        return;
    }

    for (int d = 0; d <= max_digit; d++) {
        a[pos] = d;

        int total_sum = 0;
        for (int i = 0; i <= pos; i++) {
            total_sum += a[i];
        }

        if (skip_all_pruning == 0) {
            if (total_sum > var_to_not_exceed) continue;
            if (n_plus_two == 1) {
                if (total_sum > N + 2) continue;
            }
        }

        if (pos == N * N - 1) {
            int min_row_sum = N * N * max_digit;
            int max_row_sum = -1;

            for (int r = 0; r < N; r++) {
                int current_row_sum = 0;
                for (int c = 0; c < N; c++) {
                    current_row_sum += a[r * N + c];
                }
                if (current_row_sum < min_row_sum) min_row_sum = current_row_sum;
                if (current_row_sum > max_row_sum) max_row_sum = current_row_sum;
            }

            if (skip_all_pruning == 0) {
                if (min_row_sum > upper_bound) continue;
                if (max_row_sum < lower_bound) continue;
            }
        }

        if (pos % N == N - 1) {
            int row = pos / N;
            int has_one = 0;
            for (int i = row * N; i <= pos; i++) {
                if (a[i] == 1) { has_one = 1; break; }
            }
            if (skip_all_pruning == 0) {if (!has_one) continue;}
        }

        if (no_a_to_b_squared == 1) {
            if (pos >= N * (N - 1)) {
                int col = pos % N;
                int nonzero_count = 0;
                int only_value = 0;

                for (int row = 0; row < N; row++) {
                    int val = a[row * N + col];
                    if (val != 0) {
                        nonzero_count++;
                        only_value = val;
                    }
                }

                if (nonzero_count == 1 && only_value != 1 && skip_all_pruning == 0) {
                    continue;
                }
            }
        }
        search(a, pos + 1, max_digit, work, lwork, tl, tid);
    }
}

int main(void) {
    wallStart = omp_get_wtime();
    printf("On your marks ... Get set ... Go!\n");

    var_to_not_exceed = pow(upper_bound, N) + N - 1;

    double init_work;
    LAPACKE_dgeev_work(LAPACK_ROW_MAJOR, 'N', 'N', N, NULL, N, NULL, NULL,
                       NULL, N, NULL, N, &init_work, -1);
    int lwork = (int)init_work;

    int num_threads = omp_get_max_threads();
    int base = max_digit + 1;
    int total_tasks = (int)pow(base, N);

    // Master array to hold pointers to each thread's local results for the final merge
    ThreadLocal** all_thread_results = (ThreadLocal**)calloc(num_threads, sizeof(ThreadLocal*));

    /* OPEN THE PARALLEL REGION EARLY:
       This forces each thread to allocate its own memory directly on its physical NUMA socket. */
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();

        // 1. Thread-Local State Allocation (Aligned to prevent false sharing)
        ThreadLocal* tl = NULL;
        if (posix_memalign((void**)&tl, 64, sizeof(ThreadLocal)) != 0) {
            tl = (ThreadLocal*)malloc(sizeof(ThreadLocal)); // Fallback
        }
        memset(tl, 0, sizeof(ThreadLocal));
        tl->eigenvalues = (double*)malloc(MAX_UNIQUE_EIGENVALUES * sizeof(double));

        all_thread_results[tid] = tl; // Store pointer for main thread to collect later

        // 2. Localized LAPACK Workspace Allocation
        double* private_work = (double*)malloc(lwork * sizeof(double));

        // 3. Dynamic Chunking = 1 to handle the brutal variance of the matrix tree branches
        #pragma omp for schedule(dynamic, 1)
        for (int task = 0; task < total_tasks; task++) {
            int a[N * N] = {0};

            int temp = task;
            for (int pos = 0; pos < N; pos++) {
                a[pos] = temp % base;
                temp /= base;
            }

            if (skip_all_pruning == 0) {


                int current_row_sum = 0;
                for (int c = 0; c < N; c++) {
                    current_row_sum += a[c];
                }
                if (current_row_sum > var_to_not_exceed) continue;

                int has_one = 0;


                for (int i = 0; i < N; i++) {
                    if (a[i] == 1) { has_one = 1; break; }
                }
                if (!has_one) continue;

                int has_nondiag = 0;
                for (int frpos = 1; frpos < N; frpos++) {
                    if (a[frpos] != 0) {
                        has_nondiag = 1;
                        break;
                    }
                }
                if (!has_nondiag) continue;
            }

            // Pass the localized struct pointer directly down the stack
            search(a, N, max_digit, private_work, lwork, tl, tid);
        }

        // Clean up the localized workspace (The ThreadLocal results survive until the merge)
        free(private_work);
    }

    // --------------------------------------------------------
    // MERGE PHASE
    // --------------------------------------------------------
    int total_increment = 0;
    int total_eigenwaste = 0;
    long long global_visited = 0;

    for (int t = 0; t < num_threads; t++) {
        ThreadLocal* tl = all_thread_results[t];
        if (!tl) continue; // The AI wrote "safety check"

        total_increment  += tl->increment;
        total_eigenwaste += tl->eigenwaste;
        global_visited   += tl->visited;

        for (int i = 0; i < tl->count; i++) {
            double ev = tl->eigenvalues[i];
            int found = 0;
            for (int j = 0; j < unique_count; j++) {
                if (fabs(unique_eigenvalues[j] - ev) < eigen_tolerance) {
                    found = 1;
                    break;
                }
            }
            if (!found && unique_count < MAX_UNIQUE_EIGENVALUES) {
                unique_eigenvalues[unique_count] = ev;
                unique_count++;
            }
        }
        free(tl->eigenvalues);
        free(tl);
    }
    free(all_thread_results);

    double waste_percentage = (double)100 * total_increment / (total_eigenwaste + total_increment);

    printf("\n--- Search complete ---\n");
    printf("Total matrices evaluated: ~%lld\n", global_visited);
    if (skip_all_pruning == 0) {
        printf("%d matrices valid matching criteria!\n", total_increment);
        printf("EIGENWASTE: only %d matrices! Efficiency: %.5f%%", total_eigenwaste, waste_percentage);
    }
    else {printf("And that's it since you turned pruning off!\n");}

    if (mode >= 2) {
        qsort(unique_eigenvalues, unique_count, sizeof(double), compare_doubles);
        printf("\nDiscovered %d unique eigenvalues:\n", unique_count);
        for (int i = 0; i < unique_count; i++) {
            printf("%.5f\n", unique_eigenvalues[i]);
        }
    }

    double elapsed = omp_get_wtime() - wallStart;
    printf("\nTime: %.5f s (across %d threads)\n", elapsed, num_threads);

    return 0;
}