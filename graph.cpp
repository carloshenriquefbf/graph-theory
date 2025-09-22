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

// ============================================================================
// GRAPH REPRESENTATIONS
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

        dfsRecursive(startVertex, -1, 0);
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
public:
    DiameterAlgorithm(const GraphType* g) : DistanceAlgorithm<GraphType>(g) {}

    int getDiameter() {
        int diameter = 0;
        int numVertices = this->graph->getNumVertices();

        // Otimização avançada: Para grafos muito grandes, pode pular alguns vértices
        // mas para manter a correção, faremos apenas a otimização básica

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
void runDFS(const GraphType& graph, int startVertex, const std::string& outputFilename) {
    DFSAlgorithm<GraphType> dfs(&graph);
    dfs.execute(startVertex);
    dfs.printResults(outputFilename);
    std::cout << "DFS results saved to: " << outputFilename << std::endl;
}

template <typename GraphType>
void runDiameter(const GraphType& graph, const std::string& outputFilename) {
    DiameterAlgorithm<GraphType> diameter(&graph);
    diameter.execute(1); // Start vertex doesn't matter for diameter calculation
    diameter.printResults(outputFilename);
    std::cout << "Diameter results saved to: " << outputFilename << std::endl;
}

template <typename GraphType>
void runComponents(const GraphType& graph, const std::string& outputFilename) {
    ConnectedComponentsAlgorithm<GraphType> components(&graph);
    components.execute(); // Find all components
    components.printResults(outputFilename);
    std::cout << "Connected components results saved to: " << outputFilename << std::endl;
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
    std::cout << "  adjacencyMatrix  - Use adjacency matrix representation\n";
    std::cout << "  adjacencyList    - Use adjacency list representation\n\n";
    std::cout << "Operations:\n";
    std::cout << "  stats            - Generate graph statistics only\n";
    std::cout << "  bfs <startVertex>- Run BFS from specified start vertex\n";
    std::cout << "  dfs <startVertex>- Run DFS from specified start vertex\n";
    std::cout << "  diameter         - Calculate graph diameter\n";
    std::cout << "  components       - Find connected components\n";
    std::cout << "  all <startVertex>- Run all algorithms (full analysis)\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << programName << " graph.txt adjacencyList stats\n";
    std::cout << "  " << programName << " graph.txt adjacencyMatrix bfs 1\n";
    std::cout << "  " << programName << " graph.txt adjacencyList dfs 3\n";
    std::cout << "  " << programName << " graph.txt adjacencyMatrix diameter\n";
    std::cout << "  " << programName << " graph.txt adjacencyList components\n";
    std::cout << "  " << programName << " graph.txt adjacencyMatrix all 1\n";
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        printUsage(argv[0]);
        return 1;
    }

    std::string filename = argv[1];
    std::string mode = argv[2];
    std::string operation = argv[3];

    if (mode != "adjacencyMatrix" && mode != "adjacencyList") {
        std::cerr << "Error: Invalid mode. Use 'adjacencyMatrix' or 'adjacencyList'.\n";
        return 1;
    }

    try {
        std::variant<AdjacencyMatrixGraph, AdjacencyListGraph> graph;

        if (mode == "adjacencyMatrix") {
            graph = GraphFileReader::readFromFile<AdjacencyMatrixGraph>(filename);
        } else {
            graph = GraphFileReader::readFromFile<AdjacencyListGraph>(filename);
        }

        if (operation == "stats") {
            std::string outputFilename = generateOutputFilename(filename, mode, "stats");
            std::visit([&](auto& g) { generateGraphStatistics(g, outputFilename); }, graph);
        }
        else if (operation == "bfs") {
            if (argc < 5) {
                std::cerr << "Error: BFS requires a start vertex.\n";
                return 1;
            }
            int startVertex = std::stoi(argv[4]);
            std::string outputFilename = generateOutputFilename(filename, mode, "bfs");
            std::visit([&](auto& g) { runBFS(g, startVertex, outputFilename); }, graph);
        }
        else if (operation == "dfs") {
            if (argc < 5) {
                std::cerr << "Error: DFS requires a start vertex.\n";
                return 1;
            }
            int startVertex = std::stoi(argv[4]);
            std::string outputFilename = generateOutputFilename(filename, mode, "dfs");
            std::visit([&](auto& g) { runDFS(g, startVertex, outputFilename); }, graph);
        }
        else if (operation == "diameter") {
            std::string outputFilename = generateOutputFilename(filename, mode, "diameter");
            std::visit([&](auto& g) { runDiameter(g, outputFilename); }, graph);
        }
        else if (operation == "components") {
            std::string outputFilename = generateOutputFilename(filename, mode, "components");
            std::visit([&](auto& g) { runComponents(g, outputFilename); }, graph);
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