# Graph theory - COS242

Complete graph library implementation with algorithms for BFS, DFS, distance calculation, diameter computation, and connected components analysis.

## Usage

### Compilation

To compile the program, use the following command:
```bash
make
```

Or manually:
```bash
g++ -std=c++17 -Wall -Wextra -O2 -o graph graph.cpp
```

**Note:** The entire graph library is now contained in a single `graph.cpp` file.

### Command Line Arguments

```bash
./graph <filename> <mode> <operation> [options]
```

**Parameters:**
- `filename`: Path to the input graph file
- `mode`: Graph representation (`adjacencyMatrix` or `adjacencyList`)
- `operation`: Algorithm/analysis to perform
- `[options]`: Additional parameters depending on the operation

### Available Operations

#### 1. Graph Statistics Only
```bash
./graph <filename> <mode> stats
```
Generates basic graph statistics (vertices, edges, degree information).

#### 2. BFS Algorithm
```bash
./graph <filename> <mode> bfs <startVertex>
```
Runs Breadth-First Search from the specified start vertex.

#### 3. DFS Algorithm
```bash
./graph <filename> <mode> dfs <startVertex>
```
Runs Depth-First Search from the specified start vertex.

#### 4. Graph Diameter
```bash
./graph <filename> <mode> diameter
```
Calculates the diameter of the graph (longest shortest path).

#### 5. Connected Components
```bash
./graph <filename> <mode> components
```
Finds all connected components in the graph.

#### 6. Full Analysis
```bash
./graph <filename> <mode> all <startVertex>
```
Runs all algorithms and generates comprehensive analysis.

### Examples

```bash
# Generate only graph statistics using adjacency list
./graph example.txt adjacencyList stats

# Run BFS from vertex 1 using adjacency matrix
./graph example.txt adjacencyMatrix bfs 1

# Run DFS from vertex 3 using adjacency list
./graph example.txt adjacencyList dfs 3

# Calculate graph diameter using adjacency matrix
./graph example.txt adjacencyMatrix diameter

# Find connected components using adjacency list
./graph example.txt adjacencyList components

# Run full analysis starting from vertex 1
./graph example.txt adjacencyMatrix all 1
```

### Input Format

The input file should contain graph data in the following format:

```
5
1 2
2 5
5 3
4 5
1 5
```

- First line: number of vertices
- Subsequent lines: edges (vertex pairs)

### Output Files

The program generates specific output files based on the operation:

- **stats**: `<filename>_<mode>_stats.txt`
- **bfs**: `<filename>_<mode>_bfs.txt`
- **dfs**: `<filename>_<mode>_dfs.txt`
- **diameter**: `<filename>_<mode>_diameter.txt`
- **components**: `<filename>_<mode>_components.txt`
- **all**: `<filename>_<mode>_info.txt`

### Sample Output

#### Statistics Output (`example_adjacencyList_stats.txt`)
```
GRAPH STATISTICS
================
Number of vertices: 5
Number of edges: 5
Minimum degree: 1
Maximum degree: 4
Average degree: 2.00
Median degree: 2.00
```

#### BFS Output (`example_adjacencyMatrix_bfs.txt`)
```
BFS TREE
========
Vertex | Parent | Level
-------|--------|------
     1 |   root |     0
     2 |      1 |     1
     3 |      5 |     2
     4 |      5 |     2
     5 |      1 |     1
```

#### Full Analysis Output (`example_adjacencyMatrix_info.txt`)
```
GRAPH STATISTICS
================
Number of vertices: 5
Number of edges: 5
Minimum degree: 1
Maximum degree: 4
Average degree: 2.00
Median degree: 2.00

BFS TREE
========
Vertex | Parent | Level
-------|--------|------
     1 |   root |     0
     2 |      1 |     1
     3 |      5 |     2
     4 |      5 |     2
     5 |      1 |     1

DFS TREE
========
Vertex | Parent | Level
-------|--------|------
     1 |   root |     0
     2 |      1 |     1
     3 |      5 |     3
     4 |      5 |     3
     5 |      2 |     2

GRAPH DIAMETER
==============
Diameter: 2

CONNECTED COMPONENTS
====================
Number of components: 1

Component 1 (size: 5): 1 2 5 3 4
```

## Algorithm Implementation Details

### Implementation Structure

The implementation uses a modular design with clear algorithm hierarchy:

- **`GraphAlgorithm<GraphType>`**: Base algorithm interface
- **`BFSAlgorithm<GraphType>`**: BFS implementation
- **`DFSAlgorithm<GraphType>`**: DFS implementation
- **`DistanceAlgorithm<GraphType>`**: Distance calculation using BFS
- **`DiameterAlgorithm<GraphType>`**: Diameter computation using distance methods
- **`ConnectedComponentsAlgorithm<GraphType>`**: Component discovery using DFS

### Code Reuse

The design emphasizes code reuse through modular implementation:
- **DistanceAlgorithm** reuses BFS implementation (marked with `// REUTILIZAÇÃO` comments)
- **DiameterAlgorithm** reuses DistanceAlgorithm methods
- **ConnectedComponentsAlgorithm** reuses DFS implementation

### Template Design

All algorithms are template-based, allowing them to work with both:
- `AdjacencyListGraph`
- `AdjacencyMatrixGraph`

This provides flexibility while maintaining type safety and performance optimization for different graph representations.

## File Structure

The project consists of a single unified file:
- **`graph.cpp`** - Complete graph library containing all classes and algorithms

This unified approach makes the library easy to:
- Use with specific operations for targeted analysis
- Integrate into automated testing pipelines
- Scale for large graph analysis workflows
- Maintain and extend with new algorithms