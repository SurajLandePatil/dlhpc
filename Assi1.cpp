#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <omp.h>

class Graph {
    int V;
    std::vector<std::vector<int>> adj;

public:
    Graph(int V) : V(V), adj(V) {}

    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u); // undirected graph
    }

    void parallelBFS(int start) {
        std::vector<bool> visited(V, false);
        std::queue<int> q;

        visited[start] = true;
        q.push(start);

        std::cout << "==========================\n";
        std::cout << "Parallel BFS starting from node " << start << ":\n";

        while (!q.empty()) {
            int size = q.size();

            #pragma omp parallel for
            for (int i = 0; i < size; ++i) {
                int node;
                #pragma omp critical
                {
                    node = q.front();
                    q.pop();
                }

                std::cout << "Visited node " << node << " by thread " << omp_get_thread_num() << "\n";

                for (int neighbor : adj[node]) {
                    if (!visited[neighbor]) {
                        #pragma omp critical
                        {
                            if (!visited[neighbor]) {
                                visited[neighbor] = true;
                                q.push(neighbor);
                            }
                        }
                    }
                }
            }
        }
        std::cout << "==========================\n";
    }

    void parallelDFS(int start) {
        std::vector<bool> visited(V, false);
        std::stack<int> s;

        s.push(start);

        std::cout << "Parallel DFS starting from node " << start << ":\n";

        #pragma omp parallel
        {
            while (true) {
                int node = -1;

                #pragma omp critical
                {
                    if (!s.empty()) {
                        node = s.top();
                        s.pop();
                    }
                }

                if (node == -1) break;

                if (!visited[node]) {
                    #pragma omp critical
                    {
                        if (!visited[node]) {
                            visited[node] = true;
                            std::cout << "Visited node " << node << " by thread " << omp_get_thread_num() << "\n";
                        }
                    }

                    #pragma omp critical
                    {
                        for (int neighbor : adj[node]) {
                            if (!visited[neighbor]) {
                                s.push(neighbor);
                            }
                        }
                    }
                }
            }
        }
        std::cout << "==========================\n";
    }
};

int main() {
    Graph g(7);
    g.addEdge(0, 1);
    g.addEdge(0, 2);
    g.addEdge(1, 3);
    g.addEdge(1, 4);
    g.addEdge(2, 5);
    g.addEdge(2, 6);

    g.parallelBFS(0);
    g.parallelDFS(0);

    return 0;
}
