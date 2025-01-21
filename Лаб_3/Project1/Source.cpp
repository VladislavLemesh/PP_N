#include <assert.h>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <random>
#include <vector>

using namespace std;

void mul_matrix(double* A, size_t cA, size_t rA,
    const double* B, size_t cB, size_t rB,
    const double* C, size_t cC, size_t rC)
{
    assert(cB == rC && cA == cC && rA == rB);
    assert((cA & 0x3f) == 0);

    for (size_t i = 0; i < cA; i++)
    {
        for (size_t j = 0; j < rA; j++)
        {
            A[i * rA + j] = 0;
            for (size_t k = 0; k < cB; k++)
            {
                A[i * rA + j] += B[k * rB + j] * C[i * rC + k];
            }
        }
    }
}

void mul_matrix_avx512(double* A,
    size_t cA, size_t rA,
    const double* B,
    size_t cB, size_t rB,
    const double* C,
    size_t cC, size_t rC)
{
    assert(cB == rC && cA == cC && rA == rB);
    assert((cA & 0x3f) == 0);

    for (size_t i = 0; i < rB / 8; i++)
    {
        for (size_t j = 0; j < cC; j++)
        {
            __m512d sum = _mm512_setzero_pd();
            for (size_t k = 0; k < rC; k++)
            {
                __m512d bCol = _mm512_loadu_pd(B + rB * k + i * 8);
                __m512d broadcasted = _mm512_set1_pd(C[j * rC + k]);
                sum = _mm512_fmadd_pd(bCol, broadcasted, sum);
            }

            _mm512_storeu_pd(A + j * rA + i * 8, sum);
        }
    }
}

void mul_matrix_avx2(double* A,
    size_t cA, size_t rA,
    const double* B,
    size_t cB, size_t rB,
    const double* C,
    size_t cC, size_t rC)
{
    assert(cB == rC && cA == cC && rA == rB);
    assert((cA & 0x3f) == 0);

    for (size_t i = 0; i < rB / 4; i++)
    {
        for (size_t j = 0; j < cC; j++)
        {
            __m256d sum = _mm256_setzero_pd();
            for (size_t k = 0; k < rC; k++)
            {
                __m256d bCol = _mm256_loadu_pd(B + rB * k + i * 4);
                __m256d broadcasted = _mm256_set1_pd(C[j * rC + k]);
                sum = _mm256_fmadd_pd(bCol, broadcasted, sum);
            }

            _mm256_storeu_pd(A + j * rA + i * 4, sum);
        }
    }
}

// матрица перестановок
vector<double> generate_permutation_matrix(std::size_t n)
{
    vector<double> permut_matrix(n * n, 0);

    for (std::size_t i = 0; i < n; i++)
    {
        permut_matrix[(i + 1) * n - 1 - i] = 1;
    }

    return permut_matrix;
}

int main(int argc, char** argv)
{
    const std::size_t exp_count = 10;

    std::ofstream output("output.csv");
    if (!output.is_open())
    {
        std::cout << "Error!\n";
        return -1;
    }

    auto show_matrix = [](const double* A, std::size_t colsc, std::size_t rowsc)
    {
        for (std::size_t r = 0; r < rowsc; ++r)
        {
            cout << "[" << A[r + 0 * rowsc];
            for (std::size_t c = 1; c < colsc; ++c)
            {
                cout << ", " << A[r + c * rowsc];
            }
            cout << "]\n";
        }
        cout << "\n";
    };

    auto randomize_matrix = [](double* matrix, std::size_t matrix_order) {
        std::uniform_real_distribution<double> unif(0, 100000);
        std::default_random_engine re;
        for (std::size_t i = 0; i < matrix_order * matrix_order; i++)
        {
            matrix[i] = unif(re);
        }
    };

    const std::size_t matrix_order = 16 * 4 * 9;

    vector<double> A(matrix_order * matrix_order),
        C(matrix_order * matrix_order),
        D(matrix_order * matrix_order),
        E(matrix_order * matrix_order);
    vector<double> B = generate_permutation_matrix(matrix_order);

    randomize_matrix(A.data(), matrix_order);

    std::cout << "==Correctness test. ";
    mul_matrix(C.data(), matrix_order, matrix_order,
        A.data(), matrix_order, matrix_order,
        B.data(), matrix_order, matrix_order);

    mul_matrix_avx2(D.data(), matrix_order, matrix_order,
        A.data(), matrix_order, matrix_order,
        B.data(), matrix_order, matrix_order);

    /*
    mul_matrix_avx512(E.data(), matrix_order, matrix_order,
        A.data(), matrix_order, matrix_order,
        B.data(), matrix_order, matrix_order);
    */

    if (memcmp(static_cast<void*>(C.data()),
        static_cast<void*>(D.data()),
        matrix_order * matrix_order * sizeof(double)))
    {
        cout << "FAILURE==\n";
        output.close();
        return -1;
    }

    if (memcmp(static_cast<void*>(C.data()),
        static_cast<void*>(E.data()),
        matrix_order * matrix_order * sizeof(double)))
    {
        cout << "FAILURE==\n";
        output.close();
        return -1;
    }

    cout << "ok.==\n";

    std::cout << "==Performance tests.==\n";
    output << "Type,Duration,Acceleration\n";

    double time_avg_1 = 0;
    for (std::size_t i = 0; i < exp_count; i++)
    {
        randomize_matrix(A.data(), matrix_order);
        auto t1 = std::chrono::steady_clock::now();
        mul_matrix(C.data(), matrix_order, matrix_order,
            A.data(), matrix_order, matrix_order,
            B.data(), matrix_order, matrix_order);
        auto t2 = std::chrono::steady_clock::now();
        time_avg_1 += std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    }
    time_avg_1 /= exp_count;

    std::cout << "scalar multiplication: duration = " << time_avg_1 << "ms, acceleration = 1\n";
    output << "Scalar," << time_avg_1 << ",1\n";

    double time_avg_2 = 0;
    for (std::size_t i = 0; i < exp_count; i++)
    {
        randomize_matrix(A.data(), matrix_order);
        auto t1 = std::chrono::steady_clock::now();
        mul_matrix_avx2(D.data(), matrix_order, matrix_order,
            A.data(), matrix_order, matrix_order,
            B.data(), matrix_order, matrix_order);
        auto t2 = std::chrono::steady_clock::now();
        time_avg_2 += std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    }
    time_avg_2 /= exp_count;

    std::cout << "vectorized multiplication: duration = " << time_avg_2
        << "ms, acceleration = " << time_avg_1 / time_avg_2 << "\n";
    output << "Vectorized," << time_avg_2 << "," << time_avg_1 / time_avg_2 << "\n";

    /*
    double time_avg_3 = 0;
    for (std::size_t i = 0; i < exp_count; i++)
    {
        randomize_matrix(A.data(), matrix_order);
        auto t1 = std::chrono::steady_clock::now();
        mul_matrix_avx512(D.data(), matrix_order, matrix_order,
            A.data(), matrix_order, matrix_order,
            B.data(), matrix_order, matrix_order);
        auto t2 = std::chrono::steady_clock::now();
        time_avg_3 += std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    }
    time_avg_3 /= exp_count;

    std::cout << "vectorized avx512 multiplication: duration = " << time_avg_3
        << "ms, acceleration = " << time_avg_1 / time_avg_3 << "\n";
    output << "VectorizedAVX512," << time_avg_3 << "," << time_avg_1 / time_avg_3 << "\n";
    */
    std::cout << "==Done.==\n";

    output.close();
    return 0;
}