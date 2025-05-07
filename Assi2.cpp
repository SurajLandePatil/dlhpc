#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <omp.h>

using namespace std;

// Generate random data
void generateRandomArray(vector<int>& arr, int size) {
    arr.resize(size);
    for (int& x : arr) {
        x = rand() % 1000;  // Random numbers up to 999
    }
}

// Sequential Bubble Sort
void bubbleSortSequential(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; ++i)
        for (int j = 0; j < n - i - 1; ++j)
            if (arr[j] > arr[j + 1])
                swap(arr[j], arr[j + 1]);
}

// Parallel Bubble Sort using OpenMP (odd-even transposition sort)
void bubbleSortParallel(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n; ++i) {
        int start = i % 2;
        #pragma omp parallel for
        for (int j = start; j < n - 1; j += 2) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

// Sequential Merge Sort
void merge(vector<int>& arr, int left, int mid, int right) {
    vector<int> temp(right - left + 1);
    int i = left, j = mid + 1, k = 0;
    
    while(i <= mid && j <= right) {
        if(arr[i] <= arr[j]) temp[k++] = arr[i++];
        else temp[k++] = arr[j++];
    }
    while(i <= mid) temp[k++] = arr[i++];
    while(j <= right) temp[k++] = arr[j++];

    for(int i = left, k = 0; i <= right; ++i, ++k)
        arr[i] = temp[k];
}

void mergeSortSequential(vector<int>& arr, int left, int right) {
    if(left >= right) return;
    int mid = (left + right) / 2;
    mergeSortSequential(arr, left, mid);
    mergeSortSequential(arr, mid + 1, right);
    merge(arr, left, mid, right);
}

// Parallel Merge Sort
void mergeSortParallel(vector<int>& arr, int left, int right, int depth = 0) {
    if (left >= right) return;

    if (depth >= 4) {
        // Fallback to sequential to avoid too many threads
        mergeSortSequential(arr, left, right);
        return;
    }

    int mid = (left + right) / 2;

    #pragma omp parallel sections
    {
        #pragma omp section
        mergeSortParallel(arr, left, mid, depth + 1);

        #pragma omp section
        mergeSortParallel(arr, mid + 1, right, depth + 1);
    }

    merge(arr, left, mid, right);
}

// Utility to test sort performance
void testSorting(int size) {
    vector<int> arr1, arr2, arr3, arr4;

    generateRandomArray(arr1, size);
    arr2 = arr3 = arr4 = arr1;

    double start, end;

    cout << "\n=== Sorting " << size << " elements ===\n";

    start = omp_get_wtime();
    bubbleSortSequential(arr1);
    end = omp_get_wtime();
    cout << "Sequential Bubble Sort: " << (end - start) << " seconds\n";

    start = omp_get_wtime();
    bubbleSortParallel(arr2);
    end = omp_get_wtime();
    cout << "Parallel Bubble Sort: " << (end - start) << " seconds\n";

    start = omp_get_wtime();
    mergeSortSequential(arr3, 0, arr3.size() - 1);
    end = omp_get_wtime();
    cout << "Sequential Merge Sort: " << (end - start) << " seconds\n";

    start = omp_get_wtime();
    mergeSortParallel(arr4, 0, arr4.size() - 1);
    end = omp_get_wtime();
    cout << "Parallel Merge Sort: " << (end - start) << " seconds\n";
}

int main() {
    srand(time(0));
    omp_set_num_threads(4);  // You can change the number of threads

    testSorting(1000);    // You can increase to 10,000 or 100,000 if needed
    testSorting(5000);

    return 0;
}
