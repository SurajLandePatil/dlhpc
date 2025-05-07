#include <iostream>
#include <omp.h>
#include <vector>
#include <cstdlib>
#include <ctime>
using namespace std;

void parallelReduction(const vector<int>& data) {
    int n = data.size();

    int minVal = data[0];
    int maxVal = data[0];
    long long sum = 0;

    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal) reduction(+:sum)
    for (int i = 0; i < n; ++i) {
        if (data[i] < minVal)
            minVal = data[i];
        if (data[i] > maxVal)
            maxVal = data[i];
        sum += data[i];
    }

    double avg = (double)sum / n;

    cout << "Minimum Value: " << minVal << endl;
    cout << "Maximum Value: " << maxVal << endl;
    cout << "Sum: " << sum << endl;
    cout << "Average: " << avg << endl;
}

int main() {
    const int SIZE = 1000000;  // You can change this to test performance
    vector<int> data(SIZE);

    // Initialize random data
    srand(time(0));
    for (int i = 0; i < SIZE; ++i) {
        data[i] = rand() % 10000; // random values between 0 and 9999
    }

    double start = omp_get_wtime();
    parallelReduction(data);
    double end = omp_get_wtime();

    cout << "Time taken: " << (end - start) << " seconds" << endl;
    return 0;
}
