#include <iostream>
#include <omp.h>
#include <thread>
#include <fstream>
const double N = 1000000000; /*кол во разбиений для вычисления площади*/

double f(double x)
{
    return x * x;
}

double integrate(double a, double b)
{
    double sum = 0;
    double dx = (b - a) / N;

    for (int i = 0; i < N; i++)
    {
        sum += f(a + i * dx);
    }

    return dx * sum;
}

double integrate_omp(double a, double b)
{
    double sum = 0;
    double dx = (b - a) / N;

#pragma omp parallel
    {
        unsigned t = omp_get_thread_num();
        unsigned T = omp_get_num_threads();
        double local_sum = 0;

        for (size_t i = t; i < N; i += T)
        {
            local_sum += f(a + i * dx);
        }

#pragma omp critical
        {
            sum += local_sum;
        }
    }

    return dx * sum;
}

int main()
{
    #ifdef _OPENMP
        std::cout << "OpenMP work " << _OPENMP << "\n";
    #else
        std::cout << "OpenMP not work" << "\n";
    #endif

    std::ofstream output("output.csv");
    if (!output.is_open())
    {
        std::cout << "Error!\n";
        return -1;
    }

    double t1 = omp_get_wtime();
    double result = integrate(-5, 5);
    double t2 = omp_get_wtime() - t1;
    std::cout << "integrate: T = 1, value = " << result << ", duration = " << t2 << "s, acceleration = 1\n";
    output << "T,Duration,Acceleration\n1," << t2 << ",1\n";

    double duration1 = t2;

    for (std::size_t i = 2; i <= std::thread::hardware_concurrency(); i++)
    {
        omp_set_num_threads(i);
        t1 = omp_get_wtime();
        result = integrate_omp(-5, 5);
        t2 = omp_get_wtime() - t1;

        std::cout << "integrate: T = " << i << ", value = " << result
            << ", duration = " << t2 << "s, acceleration = " << duration1 / t2 << "\n";
        output << i << "," << t2 << "," << (duration1 / t2);
        if (i < std::thread::hardware_concurrency())
        {
            output << "\n";
        }
    }

    output.close();
    return 0;
}