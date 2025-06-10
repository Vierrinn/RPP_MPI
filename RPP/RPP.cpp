#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

double calculateDeterminant(vector<vector<double>>& A, int rank, int size) {
    int n = A.size();
    double det = 1.0;

    for (int k = 0; k < n; ++k) {
        if (fabs(A[k][k]) < 1e-9) return 0; // сингул€рна

        for (int i = k + 1 + rank; i < n; i += size) {
            double f = A[i][k] / A[k][k];
            for (int j = k; j < n; ++j) {
                A[i][j] -= f * A[k][j];
            }
        }

        MPI_Bcast(&A[0][0], n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) det *= A[k][k];
    }

    return det;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size, n;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    vector<vector<double>> A;
    if (rank == 0) {
        cout << "Enter matrix size: ";
        cin >> n;
        A.resize(n, vector<double>(n));
        cout << "Enter matrix row by row:\n";
        for (auto& row : A)
            for (double& el : row)
                cin >> el;
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) A.resize(n, vector<double>(n));
    MPI_Bcast(&A[0][0], n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double start = MPI_Wtime();
    double local_det = calculateDeterminant(A, rank, size);
    double global_det = 0;
    MPI_Reduce(&local_det, &global_det, 1, MPI_DOUBLE, MPI_PROD, 0, MPI_COMM_WORLD);
    double end = MPI_Wtime();

    if (rank == 0) {
        cout << "Determinant: " << global_det << endl;
        cout << "Time: " << end - start << " seconds\n";
    }

    MPI_Finalize();
    return 0;
}
