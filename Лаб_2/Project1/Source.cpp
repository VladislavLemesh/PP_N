#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>
#include <immintrin.h>
#include <fstream>

#define cols 2048 * 2
#define rows 2048 * 2

void add_matrix(double* A, const double* B, const double* C, size_t colsc, size_t rowsc)
{
	for (size_t i = 0; i < colsc * rowsc; i++)
	{
		A[i] = B[i] + C[i];
	}
}

void add_matrix_avx(double* A, const double* B, const double* C, size_t colsc, size_t rowsc)
{
	for (size_t i = 0; i < rowsc * colsc / 4; i++)
	{
		__m256d b = _mm256_loadu_pd(&(B[i * 4]));
		__m256d c = _mm256_loadu_pd(&(C[i * 4]));
		__m256d a = _mm256_add_pd(b, c);

		_mm256_storeu_pd(&(A[i * 4]), a);
	}
}

void add_matrix_avx512(double* A, const double* B, const double* C, size_t colsc, size_t rowsc)
{
	for (size_t i = 0; i < rowsc * colsc / 8; i++)
	{
		__m512d b = _mm512_loadu_pd(&(B[i * 8]));
		__m512d c = _mm512_loadu_pd(&(C[i * 8]));
		__m512d a = _mm512_add_pd(b, c);

		_mm512_storeu_pd(&(A[i * 8]), a);
	}
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

	std::vector<double> B(cols * rows, 1), C(cols * rows, -1), A(cols * rows, 4);

	auto show_matrix = [](const double* A, size_t colsc, size_t rowsc)
	{
		for (size_t r = 0; r < rowsc; ++r)
		{
			std::cout << "[" << A[r + 0 * rowsc];
			for (size_t c = 1; c < colsc; ++c)
			{
				std::cout << ", " << A[r + c * rowsc];
			}
			std::cout << "]\n";
		}
		std::cout << "\n";
	};

	std::cout << "==Correctness test. ";
	add_matrix_avx(A.data(), B.data(), C.data(), cols, rows);
	for (std::size_t i = 0; i < cols * rows; i++)
	{
		if (A.data()[i])
		{
			std::cout << "FAILURE==\n";
			return -1;
		}
	}
	std::cout << "ok.==\n";

	/*
	std::cout << "==Correctness test. ";
	add_matrix_avx512(A.data(), B.data(), C.data(), cols, rows);
	for (std::size_t i = 0; i < cols * rows; i++)
	{
		if (A.data()[i])
		{
			std::cout << "FAILURE==\n";
			return -1;
		}
	}
	std::cout << "ok.==\n";
	*/

	std::cout << "==Performance tests.==\n";
	output << "Type,Duration,Acceleration\n";

	double time_avg_1 = 0;
	for (std::size_t i = 0; i < exp_count; i++)
	{
		auto t1 = std::chrono::steady_clock::now();
		add_matrix(A.data(), B.data(), C.data(), cols, rows);
		auto t2 = std::chrono::steady_clock::now();
		time_avg_1 += std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	}
	time_avg_1 /= exp_count;

	// show_matrix(A.data(), cols, rows);

	std::cout << "scalar addition: duration = " << time_avg_1 << "ms, acceleration = 1\n";
	output << "Scalar," << time_avg_1 << ",1\n";

	double time_avg_2 = 0;
	for (std::size_t i = 0; i < exp_count; i++)
	{
		auto t1 = std::chrono::steady_clock::now();
		add_matrix_avx(A.data(), B.data(), C.data(), cols, rows);
		auto t2 = std::chrono::steady_clock::now();
		time_avg_2 += std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	}
	time_avg_2 /= exp_count;

	std::cout << "vectorized addition: duration = " << time_avg_2
		<< "ms, acceleration = " << time_avg_1 / time_avg_2 << "\n";
	output << "Vectorized," << time_avg_2 << "," << time_avg_1 / time_avg_2 << "\n";

	/*
	double time_avg_3 = 0;
	for (std::size_t i = 0; i < exp_count; i++)
	{
		auto t1 = std::chrono::steady_clock::now();
		add_matrix_avx512(A.data(), B.data(), C.data(), cols, rows);
		auto t2 = std::chrono::steady_clock::now();
		time_avg_3 += std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	}
	time_avg_3 /= exp_count;

	std::cout << "vectorized avx512 addition: duration = " << time_avg_3
		<< "ms, acceleration = " << time_avg_1 / time_avg_3 << "\n";
	output << "VectorizedAVX512," << time_avg_3 << "," << time_avg_1 / time_avg_3 << "\n";
	*/
	std::cout << "==Done.==\n";

	output.close();
	return 0;
}