#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <algorithm>
#include <numeric>
#include <string>
#include <stdexcept>
#include <iomanip>

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
};

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
                std::cout << adjacencyMatrix[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }
};

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

    std::cout << "Statistics saved to: " << outputFilename << std::endl;
}

std::string generateOutputFilename(const std::string& inputFilename, const std::string& mode) {
    size_t lastDot = inputFilename.find_last_of(".");
    std::string baseName = (lastDot == std::string::npos) ? inputFilename : inputFilename.substr(0, lastDot);
    return baseName + "_" + mode + "_info.txt";
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <filename> <mode (adjacencyMatrix | adjacencyList)>\n";
        return 1;
    }

    std::string filename = argv[1];
    std::string mode = argv[2];

    if (mode != "adjacencyMatrix" && mode != "adjacencyList") {
        std::cerr << "Invalid mode. Use 'adjacencyMatrix' or 'adjacencyList'.\n";
        return 1;
    }
    try {
        if (mode == "adjacencyMatrix") {
            AdjacencyMatrixGraph graph = GraphFileReader::readFromFile<AdjacencyMatrixGraph>(filename);
            std::string outputFilename = generateOutputFilename(filename, mode);
            generateGraphStatistics(graph, outputFilename);
        } else {
            AdjacencyListGraph graph = GraphFileReader::readFromFile<AdjacencyListGraph>(filename);
            std::string outputFilename = generateOutputFilename(filename, mode);
            generateGraphStatistics(graph, outputFilename);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}