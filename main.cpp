#include <iostream>
#include <chrono>
#include <cmath>
#include <limits>

using namespace std;
using namespace chrono;

#define N 991
#define EPS 1e-9

bool jacobi(double ** A, double * b, double * x1, double * n, int * iterations, double * duration);
bool gauss_seidel(double ** A, double * b, double * x1, double * n, int * iterations, double * duration);
void lu_decomposition(double ** A, double * b, double * X, double * n, double * duration);
void matrix_A(double ** A, double a1, double a2, double a3);
void vector_b(double * b);

int  main( )
{
    int iterations = 0;
    double norm = 1.0;
    double duration = 0.0;
    double **A, *b, *X;

    A = new double * [N];
    b = new double [N];
    X = new double [N];

    for(int i = 0; i < N; i++) A [i] = new double [N];

    matrix_A(A, 14.0, -1.0, -1.0); // a1 = 5.0 + 9.0
    //matrix_A(A, 3.0, -1.0, -1.0);
    vector_b(b);

    cout << "Jacobi:" << endl;
    jacobi(A, b, X, &norm, &iterations, &duration);
    cout << "  Norma: " << norm << "\n  Iteracje: " << iterations << "\n  Czas: " << duration << "ms" << endl;
    cout << endl;

    cout << "Gauss-Seidel:" << endl;
    gauss_seidel(A, b, X, &norm, &iterations, &duration);
    cout << "  Norma: " << norm << "\n  Iteracje: " << iterations << "\n  Czas: " << duration << "ms" << endl;
    cout << endl;

    matrix_A(A, 3.0, -1.0, -1.0);
    cout << "LU Decopmosition:" << endl;
    lu_decomposition(A, b, X, &norm, &duration);
    cout << "  Norma: " << norm << "\n  Czas: " << duration << "ms" << endl;
    cout << endl;

    for(int i = 0; i < N; i++ ) delete [] A [i];
    delete [] A;
    delete [] b;
    delete [] X;

    return 0;
}

void residual(double ** A, double * b, double * x1, double * residuumVector){
    double sum;
    for (int i = 0; i < N; ++i) {
        sum = 0.0;
        for (int j = 0; j < N; ++j) {
            sum += A[i][j] * x1[j];
        }
        residuumVector[i] = sum;
    }
    for (int i = 0; i < N; ++i) residuumVector[i] -= b[i];
}

double norm(double * v)
{
    double n = 0;

    for (int i = 0; i < N; ++i) n += pow(v[i], 2);

    return sqrt(n);
}

bool jacobi(double ** A, double * b, double * x1, double * n, int * iterations, double * duration){

    double sum = 0;
    *n = 1.0;
    *iterations = 0;
    auto *residuumVector = new double [N];
    auto *x0 = new double [N];
    for (int i = 0; i < N; ++i) x0[i] = x1[i] = 1.0;

    auto start = chrono::high_resolution_clock::now();

    while (*n > EPS) {
        for (int i = 0; i < N; ++i)
        {
        sum = 0;

        for (int j = 0; j < N; ++j) if (j != i) sum += A[i][j] * x0[j];

        x1[i] = (b[i] - sum) / A[i][i];
        }

        for (int i = 0; i < N; ++i) x0[i] = x1[i];
        ++(*iterations);
        residual(A, b, x1, residuumVector);
        *n = norm(residuumVector);
        //cout << *n << endl;
    }

    auto end = chrono::high_resolution_clock::now();
    auto difference = end - start;
    *duration = chrono::duration<double, milli>(difference).count();

    return true;
}

bool gauss_seidel(double ** A, double * b, double * x1, double * n, int * iterations, double * duration){
    double sum = 0;
    *n = 1.0;
    *iterations = 0;
    auto *residuumVector = new double [N];
    auto *x0 = new double [N];
    for (int i = 0; i < N; ++i) x0[i] = x1[i] = 1.0;
    double inf = std::numeric_limits<double>::infinity();

    auto start = chrono::high_resolution_clock::now();

    while (*n > EPS && *n != inf) {
        for (int i = 0; i < N; ++i)
        {
            sum = 0;

            for (int j = 0; j < i; ++j) sum += A[i][j] * x1[j];
            for (int j = i+1; j < N; ++j) sum += A[i][j] * x0[j];

            x1[i] = (b[i] - sum) / A[i][i];
        }

        for (int i = 0; i < N; ++i) x0[i] = x1[i];
        ++(*iterations);
        residual(A, b, x1, residuumVector);
        *n = norm(residuumVector);
        //cout << *n << endl;
    }

    auto end = chrono::high_resolution_clock::now();
    auto difference = end - start;
    *duration = chrono::duration<double, milli>(difference).count();

    return true;
}

void lu_decomposition(double ** A, double * b, double * X, double * n, double * duration)
{
    int i, j, k;
    double s;
    auto *residuumVector = new double [N];

    auto U = new double * [N];
    for(i = 0; i < N; i++) U[i] = new double [N];
    for(i = 0; i < N; i++)
    {
        for(j = 0; j < N; j++)
        {
            U[i][j] = A[i][j];
        }
    }

    auto L = new double * [N];
    for(i = 0; i < N; i++) L[i] = new double [N];
    for(i = 0; i < N; i++)
    {
        for(j = 0; j < N; j++)
        {
            if (i==j) L[i][j] = 1;
            else L[i][j] = 0;
        }
    }

    auto start = chrono::high_resolution_clock::now();

    for (i = 0; i < N - 1; ++i)
    {
        for (j = i + 1; j < N; ++j)
        {
            L[j][i] = U[j][i] / U[i][i];

            for (k = i; k < N; ++k)
                U[j][k] = U[j][k] - L[j][i] * U[i][k];
        }
    }

    auto *y = new double [N];

    for (i = 0; i < N; ++i)
    {
        s = 0;

        for (j = 0; j < i; ++j) s += L[i][j] * y[j];

        y[i] = (b[i] - s) / L[i][i];
    }

    for (i = N - 1; i >= 0; --i)
    {
        s = 0;

        for (j = i + 1; j < N; ++j) s += U[i][j] * X[j];

        X[i] = (y[i] - s) / U[i][i];
    }

    auto end = chrono::high_resolution_clock::now();
    auto difference = end - start;
    *duration = chrono::duration<double, milli>(difference).count();

    residual(A, b, X, residuumVector);
    *n = norm(residuumVector);
}

void matrix_A(double ** A, double a1, double a2, double a3){
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            if (j == i) A[i][j] = a1;
            else if (j == i - 1 || j == i + 1) A[i][j] = a2;
            else if (j == i - 2 || j == i + 2) A[i][j] = a3;
            else A[i][j] = 0;
        }
    }
}

void vector_b(double * b){
    for (int n = 0; n < N; ++n) b[n] = sin(n * (8.0 + 1.0));
}
