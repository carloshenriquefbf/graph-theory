#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <algorithm>
#include <numeric>
#include <string>
#include <stdexcept>
#include <iomanip>
#include <queue>
#include <stack>
#include <unistd.h>
#include <sstream>
#include <mach/mach.h>
#include <chrono>
#include <variant>
#include <random>
#include <map>

// ============================================================================
// GRAPH REPRESENTATIONS
// ============================================================================

// Adjacency List Graph class
class AdjacencyListGraph {
private:
    int numVertices;
    std::vector<std::map<int, double>> adjacencyList;

public:
    AdjacencyListGraph(int vertices = 0) : numVertices(vertices) {
        adjacencyList.resize(vertices + 1);
    }

    void addEdge(int u, int v, double weight) {
        adjacencyList[u][v] = weight;
        adjacencyList[v][u] = weight;
    }

    int getNumVertices() const {
        return numVertices;
    }

    int getNumEdges() const {
        int edges = 0;
        for (int i = 1; i <= numVertices; i++) {
            edges += adjacencyList[i].size();
        }
        return edges / 2;
    }

    int getDegree(int vertex) const {
        return adjacencyList[vertex].size();
    }

    std::vector<int> getAllDegrees() const {
        std::vector<int> degrees;
        for (int i = 1; i <= numVertices; i++) {
            degrees.push_back(getDegree(i));
        }
        return degrees;
    }

    int getMinDegree() const {
        if (numVertices == 0) return 0;

        auto degrees = getAllDegrees();
        return *std::min_element(degrees.begin(), degrees.end());
    }

    int getMaxDegree() const {
        if (numVertices == 0) return 0;

        auto degrees = getAllDegrees();
        return *std::max_element(degrees.begin(), degrees.end());
    }

    double getAverageDegree() const {
        if (numVertices == 0) return 0.0;

        auto degrees = getAllDegrees();
        double sum = std::accumulate(degrees.begin(), degrees.end(), 0.0);
        return sum / numVertices;
    }

    double getMedianDegree() const {
        if (numVertices == 0) return 0.0;

        auto degrees = getAllDegrees();
        std::sort(degrees.begin(), degrees.end());

        if (degrees.size() % 2 == 0) {
            return (degrees[degrees.size()/2 - 1] + degrees[degrees.size()/2]) / 2.0;
        } else {
            return degrees[degrees.size()/2];
        }
    }

    void printAdjacencyList() const {
        for (int i = 1; i <= numVertices; i++) {
            std::cout << "Vertex " << i << ": ";
            for (const auto& [neighbor, weight] : adjacencyList[i]) {
                std::cout << neighbor << "(" << weight << ") ";
            }
            std::cout << std::endl;
        }
    }

    // Method to get neighbors for algorithms
    std::vector<int> getNeighbors(int vertex) const {
        std::vector<int> neighbors;
        for (const auto& [neighbor, weight] : adjacencyList[vertex]) {
            neighbors.push_back(neighbor);
        }
        return neighbors;
    }

    std::vector<std::pair<int, double>> getNeighborsWithWeights(int vertex) const {
        std::vector<std::pair<int, double>> neighbors;
        for (const auto& [neighbor, weight] : adjacencyList[vertex]) {
            neighbors.push_back({neighbor, weight});
        }
        return neighbors;
    }

    double getEdgeWeight(int u, int v) const {
        auto it = adjacencyList[u].find(v);
        if (it != adjacencyList[u].end()) {
            return it->second;
        }
        return std::numeric_limits<double>::infinity();
    }

    bool hasEdge(int u, int v) const {
        return adjacencyList[u].find(v) != adjacencyList[u].end();
    }

    const std::map<int, double>& getNeighborsMap(int vertex) const {
        return adjacencyList[vertex];
    }
};

// Adjacency Matrix Graph class
class AdjacencyMatrixGraph {
private:
    int numVertices;
    std::vector<std::vector<double>> adjacencyMatrix;
    static constexpr double NO_EDGE = std::numeric_limits<double>::infinity();

public:
    AdjacencyMatrixGraph(int vertices = 0) : numVertices(vertices) {
        adjacencyMatrix.resize(vertices + 1, std::vector<double>(vertices + 1, NO_EDGE));
    }

    void addEdge(int u, int v, double weight) {
        adjacencyMatrix[u][v] = weight;
        adjacencyMatrix[v][u] = weight;
    }

    int getNumVertices() const {
        return numVertices;
    }

    int getNumEdges() const {
        int edges = 0;
        for (int i = 1; i <= numVertices; i++) {
            for (int j = i + 1; j <= numVertices; j++) {
                if (adjacencyMatrix[i][j] != NO_EDGE) {
                    edges++;
                }
            }
        }
        return edges;
    }

    int getDegree(int vertex) const {
        int degree = 0;
        for (int j = 1; j <= numVertices; j++) {
            if (adjacencyMatrix[vertex][j] != NO_EDGE) {
                degree++;
            }
        }
        return degree;
    }

    std::vector<int> getAllDegrees() const {
        std::vector<int> degrees;
        for (int i = 1; i <= numVertices; i++) {
            degrees.push_back(getDegree(i));
        }
        return degrees;
    }

    int getMinDegree() const {
        if (numVertices == 0) return 0;

        auto degrees = getAllDegrees();
        return *std::min_element(degrees.begin(), degrees.end());
    }

    int getMaxDegree() const {
        if (numVertices == 0) return 0;

        auto degrees = getAllDegrees();
        return *std::max_element(degrees.begin(), degrees.end());
    }

    double getAverageDegree() const {
        if (numVertices == 0) return 0.0;

        auto degrees = getAllDegrees();
        double sum = std::accumulate(degrees.begin(), degrees.end(), 0.0);
        return sum / numVertices;
    }

    double getMedianDegree() const {
        if (numVertices == 0) return 0.0;

        auto degrees = getAllDegrees();
        std::sort(degrees.begin(), degrees.end());

        if (degrees.size() % 2 == 0) {
            return (degrees[degrees.size()/2 - 1] + degrees[degrees.size()/2]) / 2.0;
        } else {
            return degrees[degrees.size()/2];
        }
    }

    void printAdjacencyMatrix() const {
        for (int i = 1; i <= numVertices; i++) {
            std::cout << "Vertex " << i << ": ";
            for (int j = 1; j <= numVertices; j++) {
                if (adjacencyMatrix[i][j] != NO_EDGE) {
                    std::cout << adjacencyMatrix[i][j] << " ";
                } else {
                    std::cout << "inf ";
                }
            }
            std::cout << std::endl;
        }
    }

    std::vector<int> getNeighbors(int vertex) const {
        std::vector<int> neighbors;
        for (int j = 1; j <= numVertices; j++) {
            if (adjacencyMatrix[vertex][j] != NO_EDGE) {
                neighbors.push_back(j);
            }
        }
        return neighbors;
    }

    std::vector<std::pair<int, double>> getNeighborsWithWeights(int vertex) const {
        std::vector<std::pair<int, double>> neighbors;
        for (int j = 1; j <= numVertices; j++) {
            if (adjacencyMatrix[vertex][j] != NO_EDGE) {
                neighbors.push_back({j, adjacencyMatrix[vertex][j]});
            }
        }
        return neighbors;
    }

    double getEdgeWeight(int u, int v) const {
        return adjacencyMatrix[u][v];
    }

    bool hasEdge(int u, int v) const {
        return adjacencyMatrix[u][v] != NO_EDGE;
    }
};

// ============================================================================
// GRAPH ALGORITHMS
// ============================================================================

// Abstract base class for graph algorithms
template<typename GraphType>
class GraphAlgorithm {
protected:
    const GraphType* graph;
    std::vector<bool> visited;
    std::vector<int> parent;
    std::vector<int> level;

public:
    GraphAlgorithm(const GraphType* g) : graph(g) {
        if (graph) {
            visited.resize(graph->getNumVertices() + 1, false);
            parent.resize(graph->getNumVertices() + 1, -1);
            level.resize(graph->getNumVertices() + 1, -1);
        }
    }

    virtual ~GraphAlgorithm() = default;

    virtual void execute(int startVertex) = 0;
    virtual void printResults(const std::string& outputFilename) = 0;


    // Public accessors for testing
    int getParent(int vertex) const { return parent[vertex]; }
    int getLevel(int vertex) const { return level[vertex]; }
    bool isVisited(int vertex) const { return visited[vertex]; }

protected:
    virtual void reset() {
        std::fill(visited.begin(), visited.end(), false);
        std::fill(parent.begin(), parent.end(), -1);
        std::fill(level.begin(), level.end(), -1);
    }
};

// Base class for BFS algorithm
template<typename GraphType>
class BFSAlgorithm : public GraphAlgorithm<GraphType> {
protected:
    std::queue<int> bfsQueue;

public:
    BFSAlgorithm(const GraphType* g) : GraphAlgorithm<GraphType>(g) {}

    void execute(int startVertex) override {
        this->reset();

        if (startVertex < 1 || startVertex > this->graph->getNumVertices()) {
            throw std::invalid_argument("Invalid start vertex");
        }

        // Limpa a queue para garantir estado limpo
        while (!bfsQueue.empty()) {
            bfsQueue.pop();
        }

        this->visited[startVertex] = true;
        this->level[startVertex] = 0;
        this->parent[startVertex] = -1; // Root has no parent
        bfsQueue.push(startVertex);

        while (!bfsQueue.empty()) {
            int current = bfsQueue.front();
            bfsQueue.pop();

            // Get neighbors based on graph type
            std::vector<int> neighbors = getNeighbors(current);

            for (int neighbor : neighbors) {
                if (!this->visited[neighbor]) {
                    this->visited[neighbor] = true;
                    this->parent[neighbor] = current;
                    this->level[neighbor] = this->level[current] + 1;
                    bfsQueue.push(neighbor);
                }
            }
        }
    }

    void printResults(const std::string& outputFilename) override {
        std::ofstream outFile(outputFilename, std::ios::app);
        if (!outFile.is_open()) {
            throw std::runtime_error("Cannot open file: " + outputFilename);
        }

        outFile << "\nBFS TREE" << std::endl;
        outFile << "========" << std::endl;
        outFile << "Vertex | Parent | Level" << std::endl;
        outFile << "-------|--------|------" << std::endl;

        for (int i = 1; i <= this->graph->getNumVertices(); i++) {
            if (this->visited[i]) {
                outFile << std::setw(6) << i << " | ";
                if (this->parent[i] == -1) {
                    outFile << std::setw(6) << "root" << " | ";
                } else {
                    outFile << std::setw(6) << this->parent[i] << " | ";
                }
                outFile << std::setw(5) << this->level[i] << std::endl;
            }
        }
        outFile << std::endl;
    }

protected:
    std::vector<int> getNeighbors(int vertex) {
        return this->graph->getNeighbors(vertex);
    }
};

// Base class for DFS algorithm
template<typename GraphType>
class DFSAlgorithm : public GraphAlgorithm<GraphType> {
protected:
    std::stack<int> dfsStack;

public:
    DFSAlgorithm(const GraphType* g) : GraphAlgorithm<GraphType>(g) {}

    void execute(int startVertex) override {
        this->reset();

        if (startVertex < 1 || startVertex > this->graph->getNumVertices()) {
            throw std::invalid_argument("Invalid start vertex");
        }

        // Limpa o stack para garantir estado limpo
        while (!dfsStack.empty()) {
            dfsStack.pop();
        }

        dfsIterative(startVertex);
    }

    void printResults(const std::string& outputFilename) override {
        std::ofstream outFile(outputFilename, std::ios::app);
        if (!outFile.is_open()) {
            throw std::runtime_error("Cannot open file: " + outputFilename);
        }

        outFile << "\nDFS TREE" << std::endl;
        outFile << "========" << std::endl;
        outFile << "Vertex | Parent | Level" << std::endl;
        outFile << "-------|--------|------" << std::endl;

        for (int i = 1; i <= this->graph->getNumVertices(); i++) {
            if (this->visited[i]) {
                outFile << std::setw(6) << i << " | ";
                if (this->parent[i] == -1) {
                    outFile << std::setw(6) << "root" << " | ";
                } else {
                    outFile << std::setw(6) << this->parent[i] << " | ";
                }
                outFile << std::setw(5) << this->level[i] << std::endl;
            }
        }
        outFile << std::endl;
    }

protected:
    std::vector<int> getNeighbors(int vertex) {
        return this->graph->getNeighbors(vertex);
    }

private:
    [[deprecated("Use dfsIterative() instead. Recursive version causes stack overflow on large graphs.")]]
    void dfsRecursive(int vertex, int parent, int level) {
        this->visited[vertex] = true;
        this->parent[vertex] = parent;
        this->level[vertex] = level;

        std::vector<int> neighbors = getNeighbors(vertex);
        for (int neighbor : neighbors) {
            if (!this->visited[neighbor]) {
                dfsRecursive(neighbor, vertex, level + 1);
            }
        }
    }

    void dfsIterative(int startVertex) {
        // Implementação iterativa para evitar stack overflow em grafos grandes
        std::stack<std::pair<int, std::pair<int, int>>> stack; // {vertex, {parent, level}}
        stack.push({startVertex, {-1, 0}});

        while (!stack.empty()) {
            auto current = stack.top();
            stack.pop();

            int vertex = current.first;
            int parent = current.second.first;
            int level = current.second.second;

            // Verificação de bounds para evitar segmentation fault
            if (vertex < 1 || static_cast<size_t>(vertex) >= this->visited.size()) {
                continue;
            }

            if (this->visited[vertex]) continue;

            this->visited[vertex] = true;
            this->parent[vertex] = parent;
            this->level[vertex] = level;

            std::vector<int> neighbors = getNeighbors(vertex);
            for (int neighbor : neighbors) {
                if (neighbor >= 1 && static_cast<size_t>(neighbor) < this->visited.size() && !this->visited[neighbor]) {
                    stack.push({neighbor, {vertex, level + 1}});
                }
            }
        }
    }
};

// Distance algorithm that extends BFS (reuses BFS implementation)
template<typename GraphType>
class DistanceAlgorithm : public BFSAlgorithm<GraphType> {
public:
    DistanceAlgorithm(const GraphType* g) : BFSAlgorithm<GraphType>(g) {}

    int getDistance(int from, int to) {
        // REUTILIZAÇÃO: Usa a implementação de BFS para encontrar distância
        this->execute(from);

        if (!this->visited[to]) {
            return -1; // No path exists
        }

        return this->level[to];
    }

    // Otimização: BFS que calcula distâncias de um vértice para todos os outros
    std::vector<int> getAllDistancesFrom(int source) {
        this->execute(source);
        std::vector<int> distances(this->graph->getNumVertices() + 1, -1);

        for (int i = 1; i <= this->graph->getNumVertices(); i++) {
            if (this->visited[i]) {
                distances[i] = this->level[i];
            }
        }

        return distances;
    }

    void printResults(const std::string& outputFilename) override {
        // BFS tree is already printed by parent class
        // This method can be overridden to add distance-specific output
        BFSAlgorithm<GraphType>::printResults(outputFilename);
    }
};

// Diameter algorithm that extends DistanceAlgorithm (reuses distance calculation)
template<typename GraphType>
class DiameterAlgorithm : public DistanceAlgorithm<GraphType> {
private:
    int calculatedDiameter = -1;  // Store the calculated diameter
    bool isApproximate = false;   // Flag to track if approximate method was used
    int sampleSize = 0;           // Store sample size for approximate method
public:
    DiameterAlgorithm(const GraphType* g) : DistanceAlgorithm<GraphType>(g) {}

    int getDiameter() {
        int diameter = 0;
        int numVertices = this->graph->getNumVertices();

        // Otimização: Apenas O(n) BFS calls em vez de O(n²)
        // Para cada vértice, calcula distâncias para todos os outros de uma vez
        for (int i = 1; i <= numVertices; i++) {
            std::vector<int> distances = this->getAllDistancesFrom(i);

            // Encontra a maior distância a partir deste vértice
            for (int j = 1; j <= numVertices; j++) {
                if (distances[j] > diameter) {
                    diameter = distances[j];
                }
            }
        }
        calculatedDiameter = diameter;
        isApproximate = false;
        return diameter;
    }

    // Approximate diameter algorithm for large graphs
    int getApproximateDiameter(int sampleSize = 100) {
        int diameter = 0;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(1, this->graph->getNumVertices());

        for (int s = 0; s < sampleSize; s++) {
            int startVertex = dis(gen);
            this->execute(startVertex);

            // Find the farthest vertex from startVertex
            int maxDistance = 0;
            for (int i = 1; i <= this->graph->getNumVertices(); i++) {
                if (this->visited[i] && this->level[i] > maxDistance) {
                    maxDistance = this->level[i];
                }
            }

            if (maxDistance > diameter) {
                diameter = maxDistance;
            }
        }
        calculatedDiameter = diameter;
        isApproximate = true;
        this->sampleSize = sampleSize;
        return diameter;
    }

    void printResults(const std::string& outputFilename) override {
        DistanceAlgorithm<GraphType>::printResults(outputFilename);

        std::ofstream outFile(outputFilename, std::ios::app);
        if (!outFile.is_open()) {
            throw std::runtime_error("Cannot open file: " + outputFilename);
        }

        outFile << "GRAPH DIAMETER" << std::endl;
        outFile << "==============" << std::endl;
        if (calculatedDiameter == -1) {
            outFile << "Error: No diameter calculation performed!" << std::endl;
        } else if (isApproximate) {
            outFile << "Approximate Diameter (sample size: " << sampleSize << "): " << calculatedDiameter << std::endl;
        } else {
            outFile << "Diameter: " << calculatedDiameter << std::endl;
        }
    }
};

// Connected components algorithm that extends DFS (reuses DFS implementation)
template<typename GraphType>
class ConnectedComponentsAlgorithm : public DFSAlgorithm<GraphType> {
private:
    std::vector<std::vector<int>> components;

public:
    ConnectedComponentsAlgorithm(const GraphType* g) : DFSAlgorithm<GraphType>(g) {}

    void execute(int startVertex = -1) override {
        components.clear();
        this->reset();

        // If no start vertex specified, find all components
        if (startVertex == -1) {
            for (int i = 1; i <= this->graph->getNumVertices(); i++) {
                if (!this->visited[i]) {
                    std::vector<int> component;
                    findComponent(i, component);
                    if (!component.empty()) {
                        components.push_back(component);
                    }
                }
            }
        } else {
            // REUTILIZAÇÃO: Usa a implementação de DFS para componente específico
            DFSAlgorithm<GraphType>::execute(startVertex);

            // Extract component from visited vertices
            std::vector<int> component;
            for (int i = 1; i <= this->graph->getNumVertices(); i++) {
                if (this->visited[i]) {
                    component.push_back(i);
                }
            }
            if (!component.empty()) {
                components.push_back(component);
            }
        }

        // Sort components by size (descending order)
        std::sort(components.begin(), components.end(),
                  [](const std::vector<int>& a, const std::vector<int>& b) {
                      return a.size() > b.size();
                  });
    }

    void printResults(const std::string& outputFilename) override {
        std::ofstream outFile(outputFilename, std::ios::app);
        if (!outFile.is_open()) {
            throw std::runtime_error("Cannot open file: " + outputFilename);
        }

        outFile << "CONNECTED COMPONENTS" << std::endl;
        outFile << "====================" << std::endl;
        outFile << "Number of components: " << components.size() << std::endl;
        outFile << std::endl;

        for (size_t i = 0; i < components.size(); i++) {
            outFile << "Component " << (i + 1) << " (size: " << components[i].size() << "): ";
            for (size_t j = 0; j < components[i].size(); j++) {
                outFile << components[i][j];
                if (j < components[i].size() - 1) outFile << " ";
            }
            outFile << std::endl;
        }
        outFile << std::endl;
    }

private:
    void findComponent(int startVertex, std::vector<int>& component) {
        // Otimização: DFS iterativo em vez de recursivo para evitar stack overflow
        std::stack<int> stack;
        stack.push(startVertex);
        this->visited[startVertex] = true;

        while (!stack.empty()) {
            int current = stack.top();
            stack.pop();
            component.push_back(current);

            std::vector<int> neighbors = this->getNeighbors(current);
            for (int neighbor : neighbors) {
                if (!this->visited[neighbor]) {
                    this->visited[neighbor] = true;
                    stack.push(neighbor);
                }
            }
        }
    }
};

// Base class for Dijkstra algorithm
template<typename GraphType>
class DijkstraAlgorithm : public GraphAlgorithm<GraphType> {
protected:
    std::vector<double> distance;
    bool hasNegativeWeights;
    int sourceVertex;

public:
    DijkstraAlgorithm(const GraphType* g) : GraphAlgorithm<GraphType>(g), sourceVertex(-1) {
        if (g) {
            distance.resize(g->getNumVertices() + 1, std::numeric_limits<double>::infinity());
        }
        hasNegativeWeights = false;
    }

    virtual ~DijkstraAlgorithm() = default;

    bool checkNegativeWeights() {
        for (int u = 1; u <= this->graph->getNumVertices(); u++) {
            auto neighbors = this->graph->getNeighborsWithWeights(u);
            for (const auto& [v, weight] : neighbors) {
                if (weight < 0) {
                    return true;
                }
            }
        }
        return false;
    }

    double getDistance(int vertex) const {
        if (vertex < 1 || vertex > this->graph->getNumVertices()) {
            throw std::invalid_argument("Invalid vertex: " + std::to_string(vertex));
        }
        return distance[vertex];
    }

    std::vector<double> getAllDistances() const {
        return distance;
    }

    std::vector<int> getShortestPath(int destination) const {
        if (destination < 1 || destination > this->graph->getNumVertices()) {
            throw std::invalid_argument("Invalid destination vertex");
        }

        if (distance[destination] == std::numeric_limits<double>::infinity()) {
            return {}; // No path exists
        }

        std::vector<int> path;
        int current = destination;

        while (current != -1) {
            path.push_back(current);
            current = this->parent[current];
        }

        std::reverse(path.begin(), path.end());
        return path;
    }

    void printResults(const std::string& outputFilename) override {
        std::ofstream outFile(outputFilename, std::ios::app);
        if (!outFile.is_open()) {
            throw std::runtime_error("Cannot open file: " + outputFilename);
        }

        outFile << "\nDIJKSTRA" << std::endl;
        outFile << "==========================" << std::endl;

        if (hasNegativeWeights) {
            outFile << "ERROR: Graph contains negative weights!" << std::endl;
            outFile << "Dijkstra's algorithm does not support negative weights." << std::endl;
            outFile << std::endl;
            return;
        }

        outFile << "Source vertex: " << sourceVertex << std::endl;
        outFile << std::endl;
        outFile << "Vertex | Distance | Parent | Path" << std::endl;
        outFile << "-------|----------|--------|------" << std::endl;

        for (int i = 1; i <= this->graph->getNumVertices(); i++) {
            outFile << std::setw(6) << i << " | ";

            if (distance[i] == std::numeric_limits<double>::infinity()) {
                outFile << std::setw(8) << "inf" << " | ";
                outFile << std::setw(6) << "-" << " | ";
                outFile << "unreachable";
            } else {
                outFile << std::setw(8) << std::fixed << std::setprecision(2) << distance[i] << " | ";

                if (this->parent[i] == -1) {
                    outFile << std::setw(6) << "root" << " | ";
                } else {
                    outFile << std::setw(6) << this->parent[i] << " | ";
                }

                // Print path
                std::vector<int> path = getShortestPath(i);
                for (size_t j = 0; j < path.size(); j++) {
                    outFile << path[j];
                    if (j < path.size() - 1) outFile << " -> ";
                }
            }
            outFile << std::endl;
        }
        outFile << std::endl;
    }

protected:
    void reset() override {
        GraphAlgorithm<GraphType>::reset();
        std::fill(distance.begin(), distance.end(), std::numeric_limits<double>::infinity());
    }
};

template<typename GraphType>
class DijkstraVectorAlgorithm : public DijkstraAlgorithm<GraphType> {
public:
    DijkstraVectorAlgorithm(const GraphType* g) : DijkstraAlgorithm<GraphType>(g) {}

    void execute(int startVertex) override {
        if (startVertex < 1 || startVertex > this->graph->getNumVertices()) {
            throw std::invalid_argument("Invalid start vertex");
        }

        this->hasNegativeWeights = this->checkNegativeWeights();
        if (this->hasNegativeWeights) {
            return;
        }

        this->reset();
        this->sourceVertex = startVertex;

        this->distance[startVertex] = 0.0;
        this->parent[startVertex] = -1;

        // Vector to track which vertices have been processed
        std::vector<bool> processed(this->graph->getNumVertices() + 1, false);

        // Main Dijkstra loop
        for (int count = 0; count < this->graph->getNumVertices(); count++) {
            // Find vertex with minimum distance that hasn't been processed
            int u = -1;
            double minDist = std::numeric_limits<double>::infinity();

            for (int v = 1; v <= this->graph->getNumVertices(); v++) {
                if (!processed[v] && this->distance[v] < minDist) {
                    minDist = this->distance[v];
                    u = v;
                }
            }

            // If no vertex found, all remaining vertices are unreachable
            if (u == -1) break;

            processed[u] = true;
            this->visited[u] = true;

            // Update distances to neighbors
            auto neighbors = this->graph->getNeighborsWithWeights(u);
            for (const auto& [v, weight] : neighbors) {
                double newDist = this->distance[u] + weight;

                if (newDist < this->distance[v]) {
                    this->distance[v] = newDist;
                    this->parent[v] = u;
                }
            }
        }
    }

    void printResults(const std::string& outputFilename) override {
        std::ofstream outFile(outputFilename, std::ios::app);
        if (!outFile.is_open()) {
            throw std::runtime_error("Cannot open file: " + outputFilename);
        }

        outFile << "IMPLEMENTATION: Vector-based (Array)" << std::endl;
        DijkstraAlgorithm<GraphType>::printResults(outputFilename);
    }
};

// Dijkstra with heap (priority queue-based implementation)
template<typename GraphType>
class DijkstraHeapAlgorithm : public DijkstraAlgorithm<GraphType> {
private:
    struct CompareDistance {
        bool operator()(const std::pair<double, int>& a, const std::pair<double, int>& b) {
            return a.first > b.first; // Min-heap: smaller distance has higher priority
        }
    };

public:
    DijkstraHeapAlgorithm(const GraphType* g) : DijkstraAlgorithm<GraphType>(g) {}

    void execute(int startVertex) override {
        if (startVertex < 1 || startVertex > this->graph->getNumVertices()) {
            throw std::invalid_argument("Invalid start vertex");
        }

        // Check for negative weights
        this->hasNegativeWeights = this->checkNegativeWeights();
        if (this->hasNegativeWeights) {
            return; // Cannot proceed with Dijkstra
        }

        this->reset();
        this->sourceVertex = startVertex;

        // Priority queue: pair<distance, vertex>
        std::priority_queue<std::pair<double, int>,
                          std::vector<std::pair<double, int>>,
                          CompareDistance> pq;

        // Initialize source
        this->distance[startVertex] = 0.0;
        this->parent[startVertex] = -1;
        pq.push({0.0, startVertex});

        // Main Dijkstra loop with heap
        while (!pq.empty()) {
            double dist = pq.top().first;
            int u = pq.top().second;
            pq.pop();

            // Aula 20.10.2025
            // Se já processamos esse vértice com uma distância menor, ignoramos
            // (isso acontece quando inserimos múltiplas entradas para o mesmo vértice)
            if (dist > this->distance[u]) {
                continue;
            }

            this->visited[u] = true;

            // Update distances to neighbors
            auto neighbors = this->graph->getNeighborsWithWeights(u);
            for (const auto& [v, weight] : neighbors) {
                double newDist = this->distance[u] + weight;

                if (newDist < this->distance[v]) {
                    this->distance[v] = newDist;
                    this->parent[v] = u;

                    // Aula 20.10.2025
                    // Inserimos a nova distância no heap
                    // A distância antiga eventualmente será processada mas será ignorada
                    // pelo check acima (dist > this->distance[u])
                    pq.push({newDist, v});
                }
            }
        }
    }

    void printResults(const std::string& outputFilename) override {
        std::ofstream outFile(outputFilename, std::ios::app);
        if (!outFile.is_open()) {
            throw std::runtime_error("Cannot open file: " + outputFilename);
        }

        outFile << "IMPLEMENTATION: Heap-based (Priority Queue)" << std::endl;
        DijkstraAlgorithm<GraphType>::printResults(outputFilename);
    }
};

class ResearcherMapper {
private:
    std::map<int, std::string> idToName;
    std::map<std::string, int> nameToId;

public:
    void loadFromFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open researcher file: " + filename);
        }

        std::string line;
        while (std::getline(file, line)) {

            size_t commaPos = line.find(',');

            int id = std::stoi(line.substr(0, commaPos));
            std::string name = line.substr(commaPos + 1);

            idToName[id] = name;
            nameToId[name] = id;
        }

        file.close();
        std::cout << "Loaded " << idToName.size() << " researchers from " << filename << std::endl;
    }

    int getIdByName(const std::string& name) const {
        auto it = nameToId.find(name);
        if (it != nameToId.end()) {
            return it->second;
        }
        return -1;
    }

    std::string getNameById(int id) const {
        auto it = idToName.find(id);
        if (it != idToName.end()) {
            return it->second;
        }
        return "-";
    }

    bool hasResearcher(const std::string& name) const {
        return nameToId.find(name) != nameToId.end();
    }
};

// ============================================================================
// FILE READER AND UTILITIES
// ============================================================================

class GraphFileReader {
public:
    template <typename GraphType>
    static GraphType readFromFile(const std::string& filename) {
        std::ifstream file(filename);

        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }

        int numVertices;
        if (!(file >> numVertices)) {
            throw std::runtime_error("Invalid format: cannot read number of vertices");
        }

        if (numVertices <= 0) {
            throw std::invalid_argument("Number of vertices must be positive");
        }

        GraphType graph(numVertices);

        int u, v;
        float weight;
        while (file >> u >> v >> weight) {
            graph.addEdge(u, v, weight);
        }

        file.close();
        return graph;
    }
};

template <typename GraphType>
void generateGraphStatistics(const GraphType& graph, const std::string& outputFilename) {
    std::ofstream outFile(outputFilename);

    if (!outFile.is_open()) {
        throw std::runtime_error("Error creating file " + outputFilename);
    }

    outFile << std::fixed << std::setprecision(2);

    outFile << "GRAPH STATISTICS" << std::endl;
    outFile << "================" << std::endl;
    outFile << "Number of vertices: " << graph.getNumVertices() << std::endl;
    outFile << "Number of edges: " << graph.getNumEdges() << std::endl;
    outFile << "Minimum degree: " << graph.getMinDegree() << std::endl;
    outFile << "Maximum degree: " << graph.getMaxDegree() << std::endl;
    outFile << "Average degree: " << graph.getAverageDegree() << std::endl;
    outFile << "Median degree: " << graph.getMedianDegree() << std::endl;

    outFile.close();
    std::cout << "Graph statistics saved to: " << outputFilename << std::endl;
}

template <typename GraphType>
void runBFS(const GraphType& graph, int startVertex, const std::string& outputFilename) {
    BFSAlgorithm<GraphType> bfs(&graph);
    bfs.execute(startVertex);
    bfs.printResults(outputFilename);
    std::cout << "BFS results saved to: " << outputFilename << std::endl;
}

template <typename GraphType>
void runMultipleBFS(const GraphType& graph, int numTests = 100) {
    BFSAlgorithm<GraphType> bfs(&graph);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(1, graph.getNumVertices());

    for (int i = 0; i < numTests; i++) {
        int startVertex = dis(gen);
        auto startBFS = std::chrono::high_resolution_clock::now();
        bfs.execute(startVertex);
        auto endBFS = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsedBFS = endBFS - startBFS;
        std::cout << elapsedBFS.count() << "\n";
    }
}

template <typename GraphType>
void runDFS(const GraphType& graph, int startVertex, const std::string& outputFilename) {
    DFSAlgorithm<GraphType> dfs(&graph);
    dfs.execute(startVertex);
    dfs.printResults(outputFilename);
    std::cout << "DFS results saved to: " << outputFilename << std::endl;
}

template <typename GraphType>
void runMultipleDFS(const GraphType& graph, int numTests = 100) {
    DFSAlgorithm<GraphType> dfs(&graph);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(1, 10000);

    for (int i = 0; i < numTests; i++) {
        int startVertex = dis(gen);
        auto startDFS = std::chrono::high_resolution_clock::now();
        dfs.execute(startVertex);
        auto endDFS = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsedDFS = endDFS - startDFS;
        std::cout << elapsedDFS.count() << "\n";
    }
}

template <typename GraphType>
void runDiameter(const GraphType& graph, const std::string& outputFilename) {
    DiameterAlgorithm<GraphType> diameter(&graph);
    diameter.getDiameter();  // Calculate exact diameter
    diameter.execute(1); // Start vertex doesn't matter for diameter calculation
    diameter.printResults(outputFilename);
    std::cout << "Diameter results saved to: " << outputFilename << std::endl;
}

template <typename GraphType>
void runApproximateDiameter(const GraphType& graph, const std::string& outputFilename, int sampleSize = 100) {
    DiameterAlgorithm<GraphType> diameter(&graph);
    diameter.getApproximateDiameter(sampleSize);  // Calculate approximate diameter
    diameter.execute(1); // Start vertex doesn't matter for diameter calculation
    diameter.printResults(outputFilename);
    std::cout << "Approximate diameter results (sample size: " << sampleSize << ") saved to: " << outputFilename << std::endl;
}

template <typename GraphType>
void runComponents(const GraphType& graph, const std::string& outputFilename) {
    ConnectedComponentsAlgorithm<GraphType> components(&graph);
    components.execute(); // Find all components
    components.printResults(outputFilename);
    std::cout << "Connected components results saved to: " << outputFilename << std::endl;
}

template <typename GraphType>
void measureBfsDistances(const GraphType& graph) {
    DistanceAlgorithm<GraphType> bfs(&graph);
    bfs.execute(1);
    std::vector<std::pair<int, int>> vertexPairs = {{10, 20}, {10, 30}, {20, 30}};

    try {
        for (const auto& pair : vertexPairs) {
            int from = pair.first;
            int to = pair.second;
            int distance = bfs.getDistance(from, to);
            if (distance != -1) {
                std::cout << "Distance from " << from << " to " << to << " is " << distance << std::endl;
            } else {
                std::cout << "No path exists from " << from << " to " << to << std::endl;
            }
        }
    } catch (const std::exception& e) {
        std::cout << " [error: " << e.what() << "]" << std::flush;
    }
}

template <typename GraphType>
void measureDfsDistances(const GraphType& graph) {
    DistanceAlgorithm<GraphType> dfs(&graph);
    dfs.execute(1);
    std::vector<std::pair<int, int>> vertexPairs = {{10, 20}, {10, 30}, {20, 30}};

    try {
        for (const auto& pair : vertexPairs) {
            int from = pair.first;
            int to = pair.second;
            int distance = dfs.getDistance(from, to);
            if (distance != -1) {
                std::cout << "Distance from " << from << " to " << to << " is " << distance << std::endl;
            } else {
                std::cout << "No path exists from " << from << " to " << to << std::endl;
            }
        }
    } catch (const std::exception& e) {
        std::cout << " [error: " << e.what() << "]" << std::flush;
    }
}

template <typename GraphType>
void findBfsParents(const GraphType& graph) {
    BFSAlgorithm<GraphType> bfs(&graph);

    std::vector<int> startVertices = {1, 2, 3};
    std::vector<int> targetVertices = {10, 20, 30};

    try {
        for (int sv : startVertices) {
            bfs.execute(sv);
            for (int tv : targetVertices) {
                int parent = bfs.getParent(tv);
                if (parent != -1) {
                    std::cout << "Parent of " << tv << " when starting from " << sv << " is " << parent << std::endl;
                } else {
                    std::cout << tv << " has no parent when starting from " << sv << std::endl;
                }
            }
        }
    } catch (const std::exception& e) {
        std::cout << " [error: " << e.what() << "]" << std::flush;
    }
}

template <typename GraphType>
void findDfsParents(const GraphType& graph) {
    DFSAlgorithm<GraphType> dfs(&graph);

    std::vector<int> startVertices = {1, 2, 3};
    std::vector<int> targetVertices = {10, 20, 30};

    try {
        for (int sv : startVertices) {
            dfs.execute(sv);
            for (int tv : targetVertices) {
                int parent = dfs.getParent(tv);
                if (parent != -1) {
                    std::cout << "Parent of " << tv << " when starting from " << sv << " is " << parent << std::endl;
                } else {
                    std::cout << tv << " has no parent when starting from " << sv << std::endl;
                }
            }
        }
    } catch (const std::exception& e) {
        std::cout << " [error: " << e.what() << "]" << std::flush;
    }
}

template <typename GraphType>
void runDijkstraVector(const GraphType& graph, int startVertex, const std::string& outputFilename) {
    DijkstraVectorAlgorithm<GraphType> dijkstra(&graph);
    dijkstra.execute(startVertex);
    dijkstra.printResults(outputFilename);
    std::cout << "Dijkstra (vector-based) results saved to: " << outputFilename << std::endl;
}

template <typename GraphType>
void runDijkstraHeap(const GraphType& graph, int startVertex, const std::string& outputFilename) {
    DijkstraHeapAlgorithm<GraphType> dijkstra(&graph);
    dijkstra.execute(startVertex);
    dijkstra.printResults(outputFilename);
    std::cout << "Dijkstra (heap-based) results saved to: " << outputFilename << std::endl;
}

template <typename GraphType>
void measureDijkstraDistancesHeap(const GraphType& graph) {
    DijkstraHeapAlgorithm<GraphType> dijkstra(&graph);

    int startVertex = 10;
    std::vector<int> destinations = {20, 30, 40, 50, 60};

    try {
        dijkstra.execute(startVertex);

        std::cout << "DIJKSTRA (HEAP-BASED) DISTANCES AND PATHS" << std::endl;
        std::cout << "==========================================" << std::endl;
        std::cout << "Source vertex: " << startVertex << std::endl;
        std::cout << std::endl;

        for (int dest : destinations) {
            double distance = dijkstra.getDistance(dest);

            if (distance == std::numeric_limits<double>::infinity()) {
                std::cout << "No path exists from " << startVertex << " to " << dest << std::endl;
            } else {
                std::cout << "Distance from " << startVertex << " to " << dest
                         << " is " << std::fixed << std::setprecision(2) << distance << std::endl;

                std::vector<int> path = dijkstra.getShortestPath(dest);
                std::cout << "Path: ";
                for (size_t i = 0; i < path.size(); i++) {
                    std::cout << path[i];
                    if (i < path.size() - 1) std::cout << " -> ";
                }
                std::cout << std::endl << std::endl;
            }
        }
    } catch (const std::exception& e) {
        std::cout << " [error: " << e.what() << "]" << std::flush;
    }
}

template <typename GraphType>
void measureDijkstraDistancesVector(const GraphType& graph) {
    DijkstraVectorAlgorithm<GraphType> dijkstra(&graph);

    int startVertex = 10;
    std::vector<int> destinations = {20, 30, 40, 50, 60};

    try {
        dijkstra.execute(startVertex);

        std::cout << "DIJKSTRA (VECTOR-BASED) DISTANCES AND PATHS" << std::endl;
        std::cout << "============================================" << std::endl;
        std::cout << "Source vertex: " << startVertex << std::endl;
        std::cout << std::endl;

        for (int dest : destinations) {
            double distance = dijkstra.getDistance(dest);

            if (distance == std::numeric_limits<double>::infinity()) {
                std::cout << "No path exists from " << startVertex << " to " << dest << std::endl;
            } else {
                std::cout << "Distance from " << startVertex << " to " << dest
                         << " is " << std::fixed << std::setprecision(2) << distance << std::endl;

                std::vector<int> path = dijkstra.getShortestPath(dest);
                std::cout << "Path: ";
                for (size_t i = 0; i < path.size(); i++) {
                    std::cout << path[i];
                    if (i < path.size() - 1) std::cout << " -> ";
                }
                std::cout << std::endl << std::endl;
            }
        }
    } catch (const std::exception& e) {
        std::cout << " [error: " << e.what() << "]" << std::flush;
    }
}

template <typename GraphType>
void runMultipleDijkstraVector(const GraphType& graph, int numTests = 100) {
    DijkstraVectorAlgorithm<GraphType> dijkstra(&graph);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(1, graph.getNumVertices());

    for (int i = 0; i < numTests; i++) {
        int startVertex = dis(gen);
        auto start = std::chrono::high_resolution_clock::now();
        dijkstra.execute(startVertex);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << elapsed.count() << "\n";
    }
}

template <typename GraphType>
void runMultipleDijkstraHeap(const GraphType& graph, int numTests = 100) {
    DijkstraHeapAlgorithm<GraphType> dijkstra(&graph);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(1, graph.getNumVertices());

    for (int i = 0; i < numTests; i++) {
        int startVertex = dis(gen);
        auto start = std::chrono::high_resolution_clock::now();
        dijkstra.execute(startVertex);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << elapsed.count() << "\n";
    }
}

template <typename GraphType>
void measureResearchesDistancesVector(const GraphType& graph, const ResearcherMapper& researcherMapper) {
    DijkstraVectorAlgorithm<GraphType> dijkstra(&graph);

    std::string startVertex = "Edsger W. Dijkstra";
    std::vector<std::string> targetResearchers = {
        "Alan M. Turing",
        "J. B. Kruskal",
        "Jon M. Kleinberg",
        "Éva Tardos",
        "Daniel R. Figueiredo"
    };

    try {
        int startVertexId = researcherMapper.getIdByName(startVertex);
        std::vector<int> destinations;
        for (const auto& name : targetResearchers) {
            int id = researcherMapper.getIdByName(name);
            if (id == -1) {
                std::cout << "Researcher not found: " << name << std::endl;
            }
            else {
                destinations.push_back(id);
            }
        }

        dijkstra.execute(startVertexId);

        std::cout << "DIJKSTRA (VECTOR-BASED) DISTANCES AND PATHS" << std::endl;
        std::cout << "============================================" << std::endl;
        std::cout << "Source vertex: " << startVertex << std::endl;
        std::cout << std::endl;

        for (int dest : destinations) {
            double distance = dijkstra.getDistance(dest);

            if (distance == std::numeric_limits<double>::infinity()) {
                std::cout << "No path exists from " << startVertex << " to " << researcherMapper.getNameById(dest) << std::endl;
            } else {
                std::cout << "Distance from " << startVertex << " to " << researcherMapper.getNameById(dest)
                         << " is " << std::fixed << std::setprecision(2) << distance << std::endl;

                std::vector<int> path = dijkstra.getShortestPath(dest);
                std::cout << "Path: ";
                for (size_t i = 0; i < path.size(); i++) {
                    std::cout << researcherMapper.getNameById(path[i]);
                    if (i < path.size() - 1) std::cout << " -> ";
                }
                std::cout << std::endl << std::endl;
            }
        }
    } catch (const std::exception& e) {
        std::cout << " [error: " << e.what() << "]" << std::flush;
    }
}

template <typename GraphType>
void measureResearchesDistancesHeap(const GraphType& graph, const ResearcherMapper& researcherMapper) {
    DijkstraHeapAlgorithm<GraphType> dijkstra(&graph);

    std::string startVertex = "Edsger W. Dijkstra";
    std::vector<std::string> targetResearchers = {
        "Alan M. Turing",
        "J. B. Kruskal",
        "Jon M. Kleinberg",
        "Éva Tardos",
        "Daniel R. Figueiredo"
    };

    try {
        int startVertexId = researcherMapper.getIdByName(startVertex);
        std::vector<int> destinations;
        for (const auto& name : targetResearchers) {
            int id = researcherMapper.getIdByName(name);
            if (id == -1) {
                std::cout << "Researcher not found: " << name << std::endl;
            }
            else {
                destinations.push_back(id);
            }
        }

        dijkstra.execute(startVertexId);

        std::cout << "DIJKSTRA (HEAP-BASED) DISTANCES AND PATHS" << std::endl;
        std::cout << "==========================================" << std::endl;
        std::cout << "Source vertex: " << startVertex << std::endl;
        std::cout << std::endl;

        for (int dest : destinations) {
            double distance = dijkstra.getDistance(dest);

            if (distance == std::numeric_limits<double>::infinity()) {
                std::cout << "No path exists from " << startVertex << " to " << researcherMapper.getNameById(dest) << std::endl;
            } else {
                std::cout << "Distance from " << startVertex << " to " << researcherMapper.getNameById(dest)
                         << " is " << std::fixed << std::setprecision(2) << distance << std::endl;

                std::vector<int> path = dijkstra.getShortestPath(dest);
                std::cout << "Path: ";
                for (size_t i = 0; i < path.size(); i++) {
                    std::cout << researcherMapper.getNameById(path[i]);
                    if (i < path.size() - 1) std::cout << " -> ";
                }
                std::cout << std::endl << std::endl;
            }
        }
    } catch (const std::exception& e) {
        std::cout << " [error: " << e.what() << "]" << std::flush;
    }
}

template <typename GraphType>
void runFullAnalysis(const GraphType& graph, const std::string& outputFilename, int startVertex) {
    std::ofstream outFile(outputFilename);

    if (!outFile.is_open()) {
        throw std::runtime_error("Error creating file " + outputFilename);
    }

    outFile << std::fixed << std::setprecision(2);

    outFile << "GRAPH STATISTICS" << std::endl;
    outFile << "================" << std::endl;
    outFile << "Number of vertices: " << graph.getNumVertices() << std::endl;
    outFile << "Number of edges: " << graph.getNumEdges() << std::endl;
    outFile << "Minimum degree: " << graph.getMinDegree() << std::endl;
    outFile << "Maximum degree: " << graph.getMaxDegree() << std::endl;
    outFile << "Average degree: " << graph.getAverageDegree() << std::endl;
    outFile << "Median degree: " << graph.getMedianDegree() << std::endl;

    outFile.close();

    // Execute all algorithms
    try {
        // BFS Algorithm
        BFSAlgorithm<GraphType> bfs(&graph);
        bfs.execute(startVertex);
        bfs.printResults(outputFilename);

        // DFS Algorithm
        DFSAlgorithm<GraphType> dfs(&graph);
        dfs.execute(startVertex);
        dfs.printResults(outputFilename);

        // Distance and Diameter Algorithms
        DiameterAlgorithm<GraphType> diameter(&graph);
        diameter.execute(startVertex);
        diameter.printResults(outputFilename);

        // Connected Components Algorithm
        ConnectedComponentsAlgorithm<GraphType> components(&graph);
        components.execute(); // Find all components
        components.printResults(outputFilename);

    } catch (const std::exception& e) {
        std::cerr << "Error executing algorithms: " << e.what() << std::endl;
    }

    std::cout << "Full analysis saved to: " << outputFilename << std::endl;
}

std::string generateOutputFilename(const std::string& inputFilename, const std::string& mode, const std::string& operation) {
    size_t lastDot = inputFilename.find_last_of(".");
    std::string baseName = (lastDot == std::string::npos) ? inputFilename : inputFilename.substr(0, lastDot);
    return baseName + "_" + mode + "_" + operation + ".txt";
}

// MacOS specific memory usage function
size_t getMemoryUsageBytes() {
    struct task_basic_info info;
    mach_msg_type_number_t count = TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(),
                  TASK_BASIC_INFO,
                  reinterpret_cast<task_info_t>(&info),
                  &count) == KERN_SUCCESS) {
        return info.resident_size;
    }
    return 0;
}

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " <filename> <mode> <operation> [options]\n\n";
    std::cout << "Mode:\n";
    std::cout << "  adjacencyMatrix                                         - Use adjacency matrix representation\n";
    std::cout << "  adjacencyList                                           - Use adjacency list representation\n\n";
    std::cout << "Operations:\n";
    std::cout << "  stats                                                   - Generate graph statistics only\n";
    std::cout << "  bfs <startVertex>                                       - Run BFS from specified start vertex\n";
    std::cout << "  dfs <startVertex>                                       - Run DFS from specified start vertex\n";
    std::cout << "  multipleBfs <numTests>                                  - Run BFS multiple times for performance testing. <numTests> is optional (default 100)\n";
    std::cout << "  multipleDfs <numTests>                                  - Run DFS multiple times for performance testing. <numTests> is optional (default 100)\n";
    std::cout << "  diameter                                                - Calculate graph diameter\n";
    std::cout << "  approximateDiameter <sampleSize>                        - Calculate approximate graph diameter (for large graphs. <sampleSize> is optional, default 100)\n";
    std::cout << "  components                                              - Find connected components\n";
    std::cout << "  measureBfsDistances                                     - Measure distances between specific vertex pairs using BFS\n";
    std::cout << "  measureDfsDistances                                     - Measure distances between specific vertex pairs using DFS\n";
    std::cout << "  findBfsParents                                          - Find parents of specific vertices using BFS. \n";
    std::cout << "  findDfsParents                                          - Find parents of specific vertices using DFS. \n";
    std::cout << "  dijkstraVector <startVertex>                            - Run Dijkstra's algorithm (vector-based) from specified start vertex\n";
    std::cout << "  dijkstraHeap <startVertex>                              - Run Dijkstra's algorithm (heap-based) from specified start vertex\n";
    std::cout << "  measureDijkstraDistancesHeap                            - Measure distances from vertex 10 to specific vertices using Dijkstra (heap-based)\n";
    std::cout << "  measureDijkstraDistancesVector                          - Measure distances from vertex 10 to specific vertices using Dijkstra (vector-based)\n";
    std::cout << "  multipleDijkstraVector <numTests>                       - Run Dijkstra (vector-based) multiple times for performance testing. <numTests> is optional (default 100)\n";
    std::cout << "  multipleDijkstraHeap <numTests>                         - Run Dijkstra (heap-based) multiple times for performance testing. <numTests> is optional (default 100)\n";
    std::cout << "  measureResearchesDistancesVector <researchesFilePath>   - Measure distances between specific researchers mapped from <researchesFilePath> using Dijkstra (vector-based)\n";
    std::cout << "  measureResearchesDistancesHeap  <researchesFilePath>    - Measure distances between specific researchers mapped from <researchesFilePath> using Dijkstra (heap-based)\n";
    std::cout << "  all <startVertex>                                       - Run all algorithms (full analysis)\n\n";
    std::cout << "Options:\n";
    std::cout << "  --memory                                                - Print memory usage information\n";
    std::cout << "  --timing                                                - Print graph loading time\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << programName << " graph.txt adjacencyList stats\n";
    std::cout << "  " << programName << " graph.txt adjacencyMatrix bfs 1\n";
    std::cout << "  " << programName << " graph.txt adjacencyList dfs 3 --memory\n";
    std::cout << "  " << programName << " graph.txt adjacencyMatrix diameter --memory --timing\n";
    std::cout << "  " << programName << " graph.txt adjacencyList components\n";
    std::cout << "  " << programName << " graph.txt adjacencyMatrix all 1 --memory\n";
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        printUsage(argv[0]);
        return 1;
    }

    std::string filename = argv[1];
    std::string mode = argv[2];
    std::string operation = argv[3];

    bool printMemory = false;
    bool printTiming = false;

    for (int i = 4; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--memory") {
            printMemory = true;
        } else if (arg == "--timing") {
            printTiming = true;
        }
    }

    if (mode != "adjacencyMatrix" && mode != "adjacencyList") {
        std::cerr << "Error: Invalid mode. Use 'adjacencyMatrix' or 'adjacencyList'.\n";
        return 1;
    }

    try {
        std::variant<AdjacencyMatrixGraph, AdjacencyListGraph> graph;
        auto start = std::chrono::high_resolution_clock::now();

        if (mode == "adjacencyMatrix") {
            graph = GraphFileReader::readFromFile<AdjacencyMatrixGraph>(filename);
        } else {
            graph = GraphFileReader::readFromFile<AdjacencyListGraph>(filename);
        }
        auto end = std::chrono::high_resolution_clock::now();

        if (printMemory) {
            std::cout << "Memory usage: " << getMemoryUsageBytes() << " bytes\n";
        }

        if (printTiming) {
            std::chrono::duration<double> elapsed = end - start;
            std::cout << "Time taken to read graph: " << elapsed.count() << " seconds\n";
        }

        if (operation == "stats") {
            std::string outputFilename = generateOutputFilename(filename, mode, "stats");
            std::visit([&](auto& g) { generateGraphStatistics(g, outputFilename); }, graph);
        }
        else if (operation == "bfs") {
            int startVertex = (argc > 4) ? std::stoi(argv[4]) : 1;
            std::string outputFilename = generateOutputFilename(filename, mode, "bfs");
            std::visit([&](auto& g) { runBFS(g, startVertex, outputFilename); }, graph);
        }
        else if (operation == "dfs") {
            int startVertex = (argc > 4) ? std::stoi(argv[4]) : 1;
            std::string outputFilename = generateOutputFilename(filename, mode, "dfs");
            std::visit([&](auto& g) { runDFS(g, startVertex, outputFilename); }, graph);
        }
        else if (operation == "multipleBfs") {
            int numTests = (argc > 4) ? std::stoi(argv[4]) : 100;
            std::visit([&](auto& g) { runMultipleBFS(g, numTests); }, graph);
        }
        else if (operation == "multipleDfs") {
            int numTests = (argc > 4) ? std::stoi(argv[4]) : 100;
            std::visit([&](auto& g) { runMultipleDFS(g, numTests); }, graph);
        }
        else if (operation == "diameter") {
            std::string outputFilename = generateOutputFilename(filename, mode, "diameter");
            std::visit([&](auto& g) { runDiameter(g, outputFilename); }, graph);
        }
        else if (operation == "approximateDiameter") {
            int sampleSize = (argc > 4) ? std::stoi(argv[4]) : 100;
            std::string outputFilename = generateOutputFilename(filename, mode, "approx_diameter");
            std::visit([&](auto& g) { runApproximateDiameter(g, outputFilename, sampleSize); }, graph);
        }
        else if (operation == "components") {
            std::string outputFilename = generateOutputFilename(filename, mode, "components");
            std::visit([&](auto& g) { runComponents(g, outputFilename); }, graph);
        }
        else if (operation == "measureBfsDistances") {
            std::visit([&](auto& g) { measureBfsDistances(g); }, graph);
        }
        else if (operation == "measureDfsDistances") {
            std::visit([&](auto& g) { measureDfsDistances(g); }, graph);
        }
        else if (operation == "findBfsParents") {
            std::visit([&](auto& g) { findBfsParents(g); }, graph);
        }
        else if (operation == "findDfsParents") {
            std::visit([&](auto& g) { findDfsParents(g); }, graph);
        }
        else if (operation == "dijkstraVector") {
            if (argc < 5) {
                std::cerr << "Error: Dijkstra Vector requires a start vertex.\n";
                return 1;
            }
            int startVertex = std::stoi(argv[4]);
            std::string outputFilename = generateOutputFilename(filename, mode, "dijkstra_vector");
            std::visit([&](auto& g) { runDijkstraVector(g, startVertex, outputFilename); }, graph);
        }
        else if (operation == "dijkstraHeap") {
            if (argc < 5) {
                std::cerr << "Error: Dijkstra Heap requires a start vertex.\n";
                return 1;
            }
            int startVertex = std::stoi(argv[4]);
            std::string outputFilename = generateOutputFilename(filename, mode, "dijkstra_heap");
            std::visit([&](auto& g) { runDijkstraHeap(g, startVertex, outputFilename); }, graph);
        }
        else if (operation == "measureDijkstraDistancesHeap") {
            std::visit([&](auto& g) { measureDijkstraDistancesHeap(g); }, graph);
        }
        else if (operation == "measureDijkstraDistancesVector") {
            std::visit([&](auto& g) { measureDijkstraDistancesVector(g); }, graph);
        }
        else if (operation == "multipleDijkstraVector") {
            int numTests = (argc > 4) ? std::stoi(argv[4]) : 100;
            std::visit([&](auto& g) { runMultipleDijkstraVector(g, numTests); }, graph);
        }
        else if (operation == "multipleDijkstraHeap") {
            int numTests = (argc > 4) ? std::stoi(argv[4]) : 100;
            std::visit([&](auto& g) { runMultipleDijkstraHeap(g, numTests); }, graph);
        }
        else if (operation == "measureResearchesDistancesVector") {
            ResearcherMapper researcherMapper;
            if (argc < 5) {
                std::cerr << "Error: Researches file path required.\n";
                return 1;
            }
            std::string researchesFilePath = argv[4];
            researcherMapper.loadFromFile(researchesFilePath);
            std::visit([&](auto& g) { measureResearchesDistancesVector(g, researcherMapper); }, graph);
        }
        else if (operation == "measureResearchesDistancesHeap") {
            ResearcherMapper researcherMapper;
            if (argc < 5) {
                std::cerr << "Error: Researches file path required.\n";
                return 1;
            }
            std::string researchesFilePath = argv[4];
            researcherMapper.loadFromFile(researchesFilePath);
            std::visit([&](auto& g) { measureResearchesDistancesHeap(g, researcherMapper); }, graph);
        }
        else if (operation == "all") {
            if (argc < 5) {
                std::cerr << "Error: Full analysis requires a start vertex.\n";
                return 1;
            }
            int startVertex = std::stoi(argv[4]);
            std::string outputFilename = generateOutputFilename(filename, mode, "info");
            std::visit([&](auto& g) { runFullAnalysis(g, outputFilename, startVertex); }, graph);
        }
        else {
            std::cerr << "Error: Invalid operation '" << operation << "'.\n";
            printUsage(argv[0]);
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}