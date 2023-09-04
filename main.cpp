#include <iostream>
#include <cmath>
#include <omp.h>

using namespace std;

// функци€ дл€ вычислени€ коэффициентов матрицы
void compute_coeffs(double* p, double* g, double* a, double* b, double* c, double* d, double h, int n) {
#pragma omp parallel for
    for (int i = 1; i < n - 1; i++) {
        a[i] = p[i + 1] / (h * h) - (p[i + 1] - p[i - 1]) / (2.0 * h * h);
        b[i] = -2.0 * p[i] / (h * h) + g[i];
        c[i] = p[i - 1] / (h * h) + (p[i + 1] - p[i - 1]) / (2.0 * h * h);
        d[i] = -g[i];
    }
}

// функци€ дл€ решени€ системы линейных уравнений методом якоби с использованием OpenMP
void solve_jacobi(double* x, double* a, double* b, double* c, double* d, int n, double eps) {
    double* x_prev = new double[n];
    double err = 0.0;
    int iter = 0;
    do {
        // сохран€ем предыдущее приближение
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            x_prev[i] = x[i];
        }
        // вычисл€ем новое приближение
#pragma omp parallel for
        for (int i = 1; i < n - 1; i++) {
            x[i] = (d[i] - a[i] * x_prev[i - 1] - c[i] * x_prev[i + 1]) / b[i];
        }
        // вычисл€ем максимальную ошибку
        err = 0.0;
#pragma omp parallel for reduction(+ : err)
        for (int i = 1; i < n - 1; i++) {
            double e = fabs(x[i] - x_prev[i]);
            if (e > err) {
                err = e;
            }
        }
        iter++;
    } while (err > eps && iter < 10000);
    delete[] x_prev;
}

// функци€ дл€ решени€ краевой задачи методом конечных разностей с использованием OpenMP
void solve_bvp(double* p, double* g, double a, double b, double c1, double c2, int n, double eps) {
    double h = (b - a) / (n - 1);
    double* x = new double[n];
    double* a_coeffs = new double[n];
    double* b_coeffs = new double[n];
    double* c_coeffs = new double[n];
    double* d_coeffs = new double[n];

    // инициализаци€ начальных и конечных условий
    x[0] = c1;
    x[n - 1] = c2;

    // вычисление коэффициентов матрицы
    compute_coeffs(p, g, a_coeffs, b_coeffs, c_coeffs, d_coeffs, h, n);

    // решение системы линейных уравнений методом якоби
    solve_jacobi(x, a_coeffs, b_coeffs, c_coeffs, d_coeffs, n, eps);

    // вывод результатов
    for (int i = 0; i < n; i++) {
        double xi = a + i * h;
        //cout <<"x[i]"<< xi << endl;
        //cout << "y[i]" << x[i] << endl;
        cout << "x[" << i << "] = " << xi << ", y[" << i << "] = " << x[i] << endl;
    }

    delete[] x;
    delete[] a_coeffs;
    delete[] b_coeffs;
    delete[] c_coeffs;
    delete[] d_coeffs;
}

// функци€ дл€ вычислени€ значени€ функции g(x)
double g(double x) {
    return x * (x - 5);
}

int main() {
    double a = 0.0;
    double b = 5.0;
    double p = 1.0;
    double c1 = 1.0;
    double c2 = 2.0;
    int n = 10;
    double eps = 1e-6;

    double* p_arr = new double[n];
    double* g_arr = new double[n];

    // заполнение массивов p и g
    for (int i = 0; i < n; i++) {
        double xi = a + i * (b - a) / (n - 1);
        p_arr[i] = p;
        g_arr[i] = g(xi);
    }

    // решение краевой задачи методом конечных разностей с использованием OpenMP
    solve_bvp(p_arr, g_arr, a, b, c1, c2, n, eps);

    delete[] p_arr;
    delete[] g_arr;


    return 0;
}