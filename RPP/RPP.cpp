#include <cmath>
#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

#define MASTER 0
#define FROM_MASTER 1
#define FROM_WORKER 2

double MatrixDeterminant(int nDim, std::vector<double>& matrix) {
    double det = 1.0;
    for (int k = 0; k < nDim - 1; ++k) {
        double maxElem = std::abs(matrix[k * nDim + k]);
        int maxRow = k;
        for (int i = k + 1; i < nDim; ++i) {
            if (std::abs(matrix[i * nDim + k]) > maxElem) {
                maxElem = std::abs(matrix[i * nDim + k]);
                maxRow = i;
            }
        }

        if (maxRow != k) {
            for (int i = k; i < nDim; ++i)
                std::swap(matrix[k * nDim + i], matrix[maxRow * nDim + i]);
            det *= -1.0;
        }

        if (matrix[k * nDim + k] == 0.0) return 0.0;

        for (int j = k + 1; j < nDim; ++j) {
            double factor = -matrix[j * nDim + k] / matrix[k * nDim + k];
            for (int i = k; i < nDim; ++i) {
                matrix[j * nDim + i] += factor * matrix[k * nDim + i];
            }
        }
    }

    for (int i = 0; i < nDim; ++i)
        det *= matrix[i * nDim + i];

    return det;
}

double Partition(const std::vector<std::vector<double>>& matrix, int s, int end, int n) {
    double det = 0.0;
    for (int j1 = s; j1 < end; ++j1) {
        std::vector<std::vector<double>> minor(n - 1, std::vector<double>(n - 1));
        for (int i = 1; i < n; ++i) {
            int j2 = 0;
            for (int j = 0; j < n; ++j) {
                if (j == j1) continue;
                minor[i - 1][j2++] = matrix[i][j];
            }
        }

        std::vector<double> flatMinor((n - 1) * (n - 1));
        for (int i = 0; i < n - 1; ++i)
            for (int j = 0; j < n - 1; ++j)
                flatMinor[i * (n - 1) + j] = minor[i][j];

        det += std::pow(-1.0, 1.0 + j1 + 1.0) * matrix[0][j1] * MatrixDeterminant(n - 1, flatMinor);
    }

    return det;
}

int main(int argc, char* argv[]) {
    int numtasks, rank, n, offset;
    double startTime, endTime, detPart, detTotal;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    if (rank == MASTER) {
        std::cout << "Enter the size of the matrix (n x n): ";
        std::cin >> n;

        std::vector<std::vector<double>> matrix(n, std::vector<double>(n));
        srand(static_cast<unsigned>(time(nullptr)));

        for (auto& row : matrix)
            for (auto& val : row)
                val = static_cast<double>(rand() % 20 + 1);

        std::vector<double> buffer(n * n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                buffer[i * n + j] = matrix[i][j];

        offset = (n + numtasks - 1) / numtasks;

        startTime = MPI_Wtime();
        for (int dest = 1; dest < numtasks; ++dest) {
            MPI_Send(&n, 1, MPI_INT, dest, FROM_MASTER, MPI_COMM_WORLD);
            MPI_Send(buffer.data(), n * n, MPI_DOUBLE, dest, FROM_MASTER, MPI_COMM_WORLD);
        }

        detTotal = Partition(matrix, 0, offset, n);
        for (int i = 1; i < numtasks; ++i) {
            MPI_Recv(&detPart, 1, MPI_DOUBLE, i, FROM_WORKER, MPI_COMM_WORLD, &status);
            detTotal += detPart;
        }

        endTime = MPI_Wtime();
        std::cout << "Determinant: " << detTotal << std::endl;
        std::cout << "Time elapsed: " << endTime - startTime << " seconds\n";
    }
    else {
        MPI_Recv(&n, 1, MPI_INT, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
        std::vector<double> buffer(n * n);
        MPI_Recv(buffer.data(), n * n, MPI_DOUBLE, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);

        std::vector<std::vector<double>> matrix(n, std::vector<double>(n));
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                matrix[i][j] = buffer[i * n + j];

        offset = (n + numtasks - 1) / numtasks;
        int start = rank * offset;
        int end = std::min(start + offset, n);
        detPart = Partition(matrix, start, end, n);

        MPI_Send(&detPart, 1, MPI_DOUBLE, MASTER, FROM_WORKER, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
