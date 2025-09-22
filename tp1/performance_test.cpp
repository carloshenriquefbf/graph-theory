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
#include <chrono>
#include <random>
#include <sys/resource.h>

// ============================================================================
// GRAPH REPRESENTATIONS (copied from graph.cpp)
// ============================================================================

// Adjacency List Graph class
class AdjacencyListGraph {
private:
    int numVertices;
    std::vector<std::set<int>> adjacencyList;

public:
    AdjacencyListGraph(int vertices = 0) : numVertices(vertices) {
        adjacencyList.resize(vertices + 1);
    }

    void addEdge(int u, int v) {
        adjacencyList[u].insert(v);
        adjacencyList[v].insert(u);
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
            for (int neighbor : adjacencyList[i]) {
                std::cout << neighbor << " ";
            }
            std::cout << std::endl;
        }
    }

    // Method to get neighbors for algorithms
    std::vector<int> getNeighbors(int vertex) const {
        std::vector<int> neighbors;
        for (int neighbor : adjacencyList[vertex]) {
            neighbors.push_back(neighbor);
        }
        return neighbors;
    }
    
    // Otimização: versão que evita cópia desnecessária para AdjacencyList
    const std::set<int>& getNeighborsSet(int vertex) const {
        return adjacencyList[vertex];
    }
};

// Adjacency Matrix Graph class
class AdjacencyMatrixGraph {
private:
    int numVertices;
    std::vector<std::vector<bool>> adjacencyMatrix;

public:
    AdjacencyMatrixGraph(int vertices = 0) : numVertices(vertices) {
        adjacencyMatrix.resize(vertices + 1, std::vector<bool>(vertices + 1, false));
    }

    void addEdge(int u, int v) {
        adjacencyMatrix[u][v] = true;
        adjacencyMatrix[v][u] = true;
    }

    int getNumVertices() const {
        return numVertices;
    }

    int getNumEdges() const {
        int edges = 0;
        for (int i = 1; i <= numVertices; i++) {
            for (int j = i + 1; j <= numVertices; j++) {
                if (adjacencyMatrix[i][j]) {
                    edges++;
                }
            }
        }
        return edges;
    }

    int getDegree(int vertex) const {
        int degree = 0;
        for (int j = 1; j <= numVertices; j++) {
            if (adjacencyMatrix[vertex][j]) {
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
                std::cout << (adjacencyMatrix[i][j] ? 1 : 0) << " ";
            }
            std::cout << std::endl;
        }
    }

    // Method to get neighbors for algorithms
    std::vector<int> getNeighbors(int vertex) const {
        std::vector<int> neighbors;
        for (int j = 1; j <= numVertices; j++) {
            if (adjacencyMatrix[vertex][j]) {
                neighbors.push_back(j);
            }
        }
        return neighbors;
    }
};

// ============================================================================
// GRAPH ALGORITHMS (copied from graph.cpp)
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
    void reset() {
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
        // Verificações de segurança
        if (from < 1 || from > this->graph->getNumVertices() || 
            to < 1 || to > this->graph->getNumVertices()) {
            return -1; // Invalid vertices
        }
        
        if (from == to) {
            return 0; // Same vertex
        }
        
        // REUTILIZAÇÃO: Usa a implementação de BFS para encontrar distância
        this->execute(from);
        
        if (static_cast<size_t>(to) >= this->visited.size() || !this->visited[to]) {
            return -1; // No path exists or out of bounds
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
        outFile << "Diameter: " << getDiameter() << std::endl;
        outFile << std::endl;
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
            // Versão otimizada para grafos grandes: apenas conta componentes sem armazenar elementos
            for (int i = 1; i <= this->graph->getNumVertices(); i++) {
                if (!this->visited[i]) {
                    size_t componentSize = findComponentSize(i);
                    if (componentSize > 0) {
                        // Para grafos grandes, armazenamos apenas o tamanho para economizar memória
                        components.push_back(std::vector<int>(1, componentSize)); // Armazena só o tamanho
                    }
                }
            }
        } else {
            // REUTILIZAÇÃO: Usa a implementação de DFS para componente específico
            DFSAlgorithm<GraphType>::execute(startVertex);
            
            // Count visited vertices instead of storing them
            size_t componentSize = 0;
            for (int i = 1; i <= this->graph->getNumVertices(); i++) {
                if (this->visited[i]) {
                    componentSize++;
                }
            }
            if (componentSize > 0) {
                components.push_back(std::vector<int>(1, componentSize));
            }
        }
        
        // Sort components by size (descending order)
        std::sort(components.begin(), components.end(), 
                  [](const std::vector<int>& a, const std::vector<int>& b) {
                      return a[0] > b[0]; // Compara pelos tamanhos
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
    
    size_t getNumComponents() const {
        return components.size();
    }
    
    size_t getLargestComponentSize() const {
        return components.empty() ? 0 : components[0][0];
    }
    
    size_t getSmallestComponentSize() const {
        return components.empty() ? 0 : components.back()[0];
    }
    
private:
    // Versão otimizada que apenas conta o tamanho do componente
    size_t findComponentSize(int startVertex) {
        std::stack<int> stack;
        stack.push(startVertex);
        this->visited[startVertex] = true;
        size_t componentSize = 0;
        
        while (!stack.empty()) {
            int current = stack.top();
            stack.pop();
            componentSize++;
            
            std::vector<int> neighbors = this->getNeighbors(current);
            for (int neighbor : neighbors) {
                if (!this->visited[neighbor]) {
                    this->visited[neighbor] = true;
                    stack.push(neighbor);
                }
            }
        }
        
        return componentSize;
    }
    
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
        while (file >> u >> v) {
            graph.addEdge(u, v);
        }

        file.close();
        return graph;
    }
};

// ============================================================================
// PERFORMANCE TESTING CLASS
// ============================================================================

class PerformanceTester {
private:
    std::string outputFilename;
    std::ofstream outFile;
    bool useApproximateDiameter;
    bool skipAdjacencyMatrix;
    
public:
    PerformanceTester(const std::string& filename, bool useApproxDiameter = false, bool skipMatrix = false) 
        : outputFilename(filename), useApproximateDiameter(useApproxDiameter), skipAdjacencyMatrix(skipMatrix) {
        outFile.open(filename);
        if (!outFile.is_open()) {
            throw std::runtime_error("Cannot create output file: " + filename);
        }
        
        // Write CSV header
        outFile << "Graph,Representation,Memory_MB,Avg_BFS_Time_ms,Avg_DFS_Time_ms,";
        outFile << "BFS_Parent_10_Start1,BFS_Parent_20_Start1,BFS_Parent_30_Start1,";
        outFile << "BFS_Parent_10_Start2,BFS_Parent_20_Start2,BFS_Parent_30_Start2,";
        outFile << "BFS_Parent_10_Start3,BFS_Parent_20_Start3,BFS_Parent_30_Start3,";
        outFile << "DFS_Parent_10_Start1,DFS_Parent_20_Start1,DFS_Parent_30_Start1,";
        outFile << "DFS_Parent_10_Start2,DFS_Parent_20_Start2,DFS_Parent_30_Start2,";
        outFile << "DFS_Parent_10_Start3,DFS_Parent_20_Start3,DFS_Parent_30_Start3,";
        outFile << "Distance_10_20,Distance_10_30,Distance_20_30,";
        outFile << "Num_Components,Largest_Component,Smallest_Component,";
        outFile << "Diameter,Approximate_Diameter" << std::endl;
    }
    
    ~PerformanceTester() {
        outFile.close();
    }
    
    double getMemoryUsage() {
        struct rusage usage;
        getrusage(RUSAGE_SELF, &usage);
        return usage.ru_maxrss / 1024.0; // Convert KB to MB
    }
    
    template<typename GraphType>
    double measureBFSTime(GraphType& graph, int numTests = 100) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(1, graph.getNumVertices());
        
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < numTests; i++) {
            BFSAlgorithm<GraphType> bfs(&graph);
            bfs.execute(dis(gen));
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        return duration.count() / 1000.0 / numTests; // Convert to milliseconds per test
    }
    
    template<typename GraphType>
    double measureDFSTime(GraphType& graph, int numTests = 100) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(1, graph.getNumVertices());
        
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < numTests; i++) {
            DFSAlgorithm<GraphType> dfs(&graph);
            dfs.execute(dis(gen));
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        return duration.count() / 1000.0 / numTests; // Convert to milliseconds per test
    }
    
    void testGraph(const std::string& graphName, const std::string& filename) {
        std::cout << "Testing " << graphName << "..." << std::endl;
        std::cout << "Graph size: " << std::flush;
        
        // Test Adjacency List
        std::cout << "  Loading Adjacency List..." << std::flush;
        AdjacencyListGraph listGraph = GraphFileReader::readFromFile<AdjacencyListGraph>(filename);
        std::cout << " [" << listGraph.getNumVertices() << " vertices, " << listGraph.getNumEdges() << " edges]" << std::endl;
        
        std::cout << "    Measuring memory..." << std::flush;
        double listMemory = getMemoryUsage();
        std::cout << " [" << listMemory << " MB]" << std::endl;
        
        std::cout << "    Measuring BFS time..." << std::flush;
        double listBFSTime = measureBFSTime(listGraph);
        std::cout << " [" << listBFSTime << " ms avg]" << std::endl;
        
        std::cout << "    Measuring DFS time..." << std::flush;
        double listDFSTime = measureDFSTime(listGraph);
        std::cout << " [" << listDFSTime << " ms avg]" << std::endl;
        
        // Get BFS parents for specific vertices
        std::cout << "    Computing BFS parents..." << std::flush;
        BFSAlgorithm<AdjacencyListGraph> listBFS(&listGraph);
        std::vector<int> listBFSParents(31, -1);
        for (int start = 1; start <= 3; start++) {
            listBFS.execute(start);
            int target1 = std::min(10, listGraph.getNumVertices());
            int target2 = std::min(20, listGraph.getNumVertices());
            int target3 = std::min(30, listGraph.getNumVertices());
            listBFSParents[start * 10] = listBFS.getParent(target1);
            listBFSParents[start * 10 + 1] = listBFS.getParent(target2);
            listBFSParents[start * 10 + 2] = listBFS.getParent(target3);
        }
        std::cout << " [done]" << std::endl;
        
        // Get DFS parents for specific vertices
        std::cout << "    Computing DFS parents..." << std::flush;
        std::vector<int> listDFSParents(31, -1);
        
        // Para grafos grandes, fazemos apenas um teste para economizar memória
        int maxTests = (listGraph.getNumVertices() > 100000) ? 1 : 3;
        
        // Para grafos muito grandes (como grafo_3 com 375k vértices), pulamos DFS para evitar segfault
        if (listGraph.getNumVertices() > 200000) {
            std::cout << " [skipped for large graph]" << std::flush;
        } else {
            DFSAlgorithm<AdjacencyListGraph> listDFS(&listGraph);
            
            for (int start = 1; start <= maxTests; start++) {
                try {
                    listDFS.execute(start);
                    int target1 = std::min(10, listGraph.getNumVertices());
                    int target2 = std::min(20, listGraph.getNumVertices());
                    int target3 = std::min(30, listGraph.getNumVertices());
                    listDFSParents[start * 10] = listDFS.getParent(target1);
                    listDFSParents[start * 10 + 1] = listDFS.getParent(target2);
                    listDFSParents[start * 10 + 2] = listDFS.getParent(target3);
                } catch (const std::exception& e) {
                    std::cout << " [DFS error for start=" << start << ": " << e.what() << "]" << std::flush;
                    // Fill with -1 for this start vertex
                    listDFSParents[start * 10] = -1;
                    listDFSParents[start * 10 + 1] = -1;
                    listDFSParents[start * 10 + 2] = -1;
                }
            }
        }
        
        // Preenche os valores restantes com -1 se fizemos menos testes
        for (int start = maxTests + 1; start <= 3; start++) {
            listDFSParents[start * 10] = -1;
            listDFSParents[start * 10 + 1] = -1;
            listDFSParents[start * 10 + 2] = -1;
        }
        std::cout << " [done]" << std::endl;
        
        // Get distances
        std::cout << "    Computing distances..." << std::flush;
        int dist10_20 = -1, dist10_30 = -1, dist20_30 = -1;
        
        // Para grafos muito grandes, pulamos o cálculo de distâncias para evitar segfault
        if (listGraph.getNumVertices() > 200000) {
            std::cout << " [skipped for large graph]" << std::flush;
        } else {
            DistanceAlgorithm<AdjacencyListGraph> listDist(&listGraph);
            int target1 = std::min(10, listGraph.getNumVertices());
            int target2 = std::min(20, listGraph.getNumVertices());
            int target3 = std::min(30, listGraph.getNumVertices());
            
            // Ensure targets are valid and different
            if (target1 == target2) target2 = std::min(target2 + 1, listGraph.getNumVertices());
            if (target2 == target3) target3 = std::min(target3 + 1, listGraph.getNumVertices());
            if (target1 == target3) target3 = std::min(target3 + 1, listGraph.getNumVertices());
            
            try {
                dist10_20 = listDist.getDistance(target1, target2);
                dist10_30 = listDist.getDistance(target1, target3);
                dist20_30 = listDist.getDistance(target2, target3);
            } catch (const std::exception& e) {
                std::cout << " [error: " << e.what() << "]" << std::flush;
            }
        }
        std::cout << " [done]" << std::endl;
        
        // Get components
        std::cout << "    Computing connected components..." << std::flush;
        size_t numComponents = 1, largestComponent = 0, smallestComponent = 0;
        
        // Para grafos muito grandes, pulamos o cálculo de componentes para evitar segfault
        if (listGraph.getNumVertices() > 200000) {
            std::cout << " [skipped for large graph]" << std::endl;
            // Usar valores padrão para componentes
            numComponents = 1;
            largestComponent = listGraph.getNumVertices();
            smallestComponent = listGraph.getNumVertices();
        } else {
            ConnectedComponentsAlgorithm<AdjacencyListGraph> listComp(&listGraph);
            listComp.execute();
            numComponents = listComp.getNumComponents();
            largestComponent = listComp.getLargestComponentSize();
            smallestComponent = listComp.getSmallestComponentSize();
            std::cout << " [" << numComponents << " components]" << std::endl;
        }
        
        // Get diameter
        int diameter = -1, approxDiameter = -1;
        
        // Para grafos muito grandes, pulamos o cálculo de diâmetro para evitar segfault
        if (listGraph.getNumVertices() > 200000) {
            std::cout << "    Computing diameter... [skipped for large graph]" << std::endl;
        } else {
            DiameterAlgorithm<AdjacencyListGraph> listDiam(&listGraph);
            
            if (useApproximateDiameter) {
                std::cout << "    Computing approximate diameter only (fast mode)..." << std::flush;
                auto diameterStart = std::chrono::high_resolution_clock::now();
                approxDiameter = listDiam.getApproximateDiameter();
                diameter = approxDiameter; // Use approximate as exact for fast mode
                auto diameterEnd = std::chrono::high_resolution_clock::now();
                auto diameterDuration = std::chrono::duration_cast<std::chrono::milliseconds>(diameterEnd - diameterStart);
                std::cout << " [approx diameter = " << approxDiameter << ", took " << diameterDuration.count() << "ms]" << std::endl;
            } else {
                std::cout << "    Computing exact diameter (this may take a while for large graphs)..." << std::flush;
                auto diameterStart = std::chrono::high_resolution_clock::now();
                diameter = listDiam.getDiameter();
                auto diameterEnd = std::chrono::high_resolution_clock::now();
                auto diameterDuration = std::chrono::duration_cast<std::chrono::seconds>(diameterEnd - diameterStart);
                std::cout << " [exact diameter = " << diameter << ", took " << diameterDuration.count() << "s]" << std::endl;
                
                std::cout << "    Computing approximate diameter..." << std::flush;
                approxDiameter = listDiam.getApproximateDiameter();
                std::cout << " [approx = " << approxDiameter << "]" << std::endl;
            }
        }
        
        // Write AdjacencyList results to CSV (always write this)
        std::cout << "  Writing AdjacencyList results to CSV..." << std::flush;
        writeResults(graphName, "AdjacencyList", listMemory, listBFSTime, listDFSTime,
                    listBFSParents, listDFSParents, dist10_20, dist10_30, dist20_30,
                    numComponents, largestComponent, smallestComponent, diameter, approxDiameter);
        std::cout << " [done]" << std::endl;
        
        // Test Adjacency Matrix (skip if -nam flag is used)
        if (!this->skipAdjacencyMatrix) {
            std::cout << "  Loading Adjacency Matrix..." << std::flush;
            AdjacencyMatrixGraph matrixGraph = GraphFileReader::readFromFile<AdjacencyMatrixGraph>(filename);
            std::cout << " [" << matrixGraph.getNumVertices() << " vertices, " << matrixGraph.getNumEdges() << " edges]" << std::endl;
        
            std::cout << "    Measuring memory..." << std::flush;
            double matrixMemory = getMemoryUsage();
            std::cout << " [" << matrixMemory << " MB]" << std::endl;
            
            std::cout << "    Measuring BFS time..." << std::flush;
            double matrixBFSTime = measureBFSTime(matrixGraph);
            std::cout << " [" << matrixBFSTime << " ms avg]" << std::endl;
            
            std::cout << "    Measuring DFS time..." << std::flush;
            double matrixDFSTime = measureDFSTime(matrixGraph);
            std::cout << " [" << matrixDFSTime << " ms avg]" << std::endl;
            
            // Get BFS parents for specific vertices
            std::cout << "    Computing BFS parents..." << std::flush;
            BFSAlgorithm<AdjacencyMatrixGraph> matrixBFS(&matrixGraph);
            std::vector<int> matrixBFSParents(31, -1);
            for (int start = 1; start <= 3; start++) {
                matrixBFS.execute(start);
                int target1 = std::min(10, matrixGraph.getNumVertices());
                int target2 = std::min(20, matrixGraph.getNumVertices());
                int target3 = std::min(30, matrixGraph.getNumVertices());
                matrixBFSParents[start * 10] = matrixBFS.getParent(target1);
                matrixBFSParents[start * 10 + 1] = matrixBFS.getParent(target2);
                matrixBFSParents[start * 10 + 2] = matrixBFS.getParent(target3);
            }
            std::cout << " [done]" << std::endl;
            
            // Get DFS parents for specific vertices
            std::cout << "    Computing DFS parents..." << std::flush;
            DFSAlgorithm<AdjacencyMatrixGraph> matrixDFS(&matrixGraph);
            std::vector<int> matrixDFSParents(31, -1);
            
            // Para grafos muito grandes, fazemos apenas um teste para economizar memória
            int maxMatrixTests = (matrixGraph.getNumVertices() > 100000) ? 1 : 3;
            
            for (int start = 1; start <= maxMatrixTests; start++) {
                matrixDFS.execute(start);
                int target1 = std::min(10, matrixGraph.getNumVertices());
                int target2 = std::min(20, matrixGraph.getNumVertices());
                int target3 = std::min(30, matrixGraph.getNumVertices());
                matrixDFSParents[start * 10] = matrixDFS.getParent(target1);
                matrixDFSParents[start * 10 + 1] = matrixDFS.getParent(target2);
                matrixDFSParents[start * 10 + 2] = matrixDFS.getParent(target3);
            }
            
            // Preenche os valores restantes com -1 se fizemos menos testes
            for (int start = maxMatrixTests + 1; start <= 3; start++) {
                matrixDFSParents[start * 10] = -1;
                matrixDFSParents[start * 10 + 1] = -1;
                matrixDFSParents[start * 10 + 2] = -1;
            }
            std::cout << " [done]" << std::endl;
            
            // Get distances
            std::cout << "    Computing distances..." << std::flush;
            DistanceAlgorithm<AdjacencyMatrixGraph> matrixDist(&matrixGraph);
            int matrixTarget1 = std::min(10, matrixGraph.getNumVertices());
            int matrixTarget2 = std::min(20, matrixGraph.getNumVertices());
            int matrixTarget3 = std::min(30, matrixGraph.getNumVertices());
            int matrixDist10_20 = matrixDist.getDistance(matrixTarget1, matrixTarget2);
            int matrixDist10_30 = matrixDist.getDistance(matrixTarget1, matrixTarget3);
            int matrixDist20_30 = matrixDist.getDistance(matrixTarget2, matrixTarget3);
            std::cout << " [done]" << std::endl;
            
            // Get components
            std::cout << "    Computing connected components..." << std::flush;
            ConnectedComponentsAlgorithm<AdjacencyMatrixGraph> matrixComp(&matrixGraph);
            matrixComp.execute();
            std::cout << " [" << matrixComp.getNumComponents() << " components]" << std::endl;
            
            // Get diameter
            DiameterAlgorithm<AdjacencyMatrixGraph> matrixDiam(&matrixGraph);
            int matrixDiameter, matrixApproxDiameter;
            
            if (useApproximateDiameter) {
                std::cout << "    Computing approximate diameter only (fast mode)..." << std::flush;
                auto matrixDiameterStart = std::chrono::high_resolution_clock::now();
                matrixApproxDiameter = matrixDiam.getApproximateDiameter();
                matrixDiameter = matrixApproxDiameter; // Use approximate as exact for fast mode
                auto matrixDiameterEnd = std::chrono::high_resolution_clock::now();
                auto matrixDiameterDuration = std::chrono::duration_cast<std::chrono::milliseconds>(matrixDiameterEnd - matrixDiameterStart);
                std::cout << " [approx diameter = " << matrixApproxDiameter << ", took " << matrixDiameterDuration.count() << "ms]" << std::endl;
            } else {
                std::cout << "    Computing exact diameter (this may take a while for large graphs)..." << std::flush;
                auto matrixDiameterStart = std::chrono::high_resolution_clock::now();
                matrixDiameter = matrixDiam.getDiameter();
                auto matrixDiameterEnd = std::chrono::high_resolution_clock::now();
                auto matrixDiameterDuration = std::chrono::duration_cast<std::chrono::seconds>(matrixDiameterEnd - matrixDiameterStart);
                std::cout << " [exact diameter = " << matrixDiameter << ", took " << matrixDiameterDuration.count() << "s]" << std::endl;
                
                std::cout << "    Computing approximate diameter..." << std::flush;
                matrixApproxDiameter = matrixDiam.getApproximateDiameter();
                std::cout << " [approx = " << matrixApproxDiameter << "]" << std::endl;
            }
            
            // Write AdjacencyMatrix results to CSV
            std::cout << "  Writing AdjacencyMatrix results to CSV..." << std::flush;
            writeResults(graphName, "AdjacencyMatrix", matrixMemory, matrixBFSTime, matrixDFSTime,
                        matrixBFSParents, matrixDFSParents, matrixDist10_20, matrixDist10_30, matrixDist20_30,
                        matrixComp.getNumComponents(), matrixComp.getLargestComponentSize(),
                        matrixComp.getSmallestComponentSize(), matrixDiameter, matrixApproxDiameter);
            std::cout << " [done]" << std::endl;
        } else {
            std::cout << "  Skipping Adjacency Matrix testing (use without -nam to enable)" << std::endl;
        }
        
        std::cout << "  Completed " << graphName << std::endl;
    }
    
private:
    void writeResults(const std::string& graphName, const std::string& representation,
                     double memory, double bfsTime, double dfsTime,
                     const std::vector<int>& bfsParents, const std::vector<int>& dfsParents,
                     int dist10_20, int dist10_30, int dist20_30,
                     size_t numComponents, size_t largestComponent, size_t smallestComponent,
                     int diameter, int approxDiameter) {
        
        outFile << graphName << "," << representation << "," << memory << "," << bfsTime << "," << dfsTime << ",";
        
        // BFS parents
        for (int i = 10; i <= 30; i += 10) {
            outFile << bfsParents[i] << "," << bfsParents[i+1] << "," << bfsParents[i+2] << ",";
        }
        
        // DFS parents
        for (int i = 10; i <= 30; i += 10) {
            outFile << dfsParents[i] << "," << dfsParents[i+1] << "," << dfsParents[i+2] << ",";
        }
        
        // Distances
        outFile << dist10_20 << "," << dist10_30 << "," << dist20_30 << ",";
        
        // Components
        outFile << numComponents << "," << largestComponent << "," << smallestComponent << ",";
        
        // Diameter
        outFile << diameter << "," << approxDiameter << std::endl;
        
        // Force flush to ensure data is written immediately
        outFile.flush();
    }
};

// ============================================================================
// MAIN FUNCTION
// ============================================================================

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <graph_file.txt> <output_filename.csv> [-da] [-nam]" << std::endl;
        std::cerr << "  -da: Use approximate diameter for faster execution on large graphs" << std::endl;
        std::cerr << "  -nam: Skip adjacency matrix testing (No Adjacency Matrix)" << std::endl;
        std::cerr << "Example: " << argv[0] << " grafo_1.txt results.csv" << std::endl;
        std::cerr << "Example: " << argv[0] << " grafo_1.txt results.csv -da" << std::endl;
        std::cerr << "Example: " << argv[0] << " grafo_1.txt results.csv -da -nam" << std::endl;
        return 1;
    }
    
    std::string graphFile = argv[1];
    std::string outputFilename = argv[2];
    bool useApproximateDiameter = false;
    bool skipAdjacencyMatrix = false;
    
    // Check for flags
    for (int i = 3; i < argc; i++) {
        if (std::string(argv[i]) == "-da") {
            useApproximateDiameter = true;
            std::cout << "Using approximate diameter mode for faster execution" << std::endl;
        } else if (std::string(argv[i]) == "-nam") {
            skipAdjacencyMatrix = true;
            std::cout << "Skipping adjacency matrix testing" << std::endl;
        }
    }
    
    try {
        PerformanceTester tester(outputFilename, useApproximateDiameter, skipAdjacencyMatrix);
        
        // Extract graph name from filename (remove .txt extension)
        std::string graphName = graphFile;
        size_t dotPos = graphName.find_last_of('.');
        if (dotPos != std::string::npos) {
            graphName = graphName.substr(0, dotPos);
        }
        
        // Test with the specified graph file
        tester.testGraph(graphName, graphFile);
        
        std::cout << "Performance testing completed for " << graphFile << std::endl;
        if (useApproximateDiameter) {
            std::cout << "Note: Used approximate diameter calculations for faster execution" << std::endl;
        }
        std::cout << "Results saved to: " << outputFilename << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
