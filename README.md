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
./graph <input_filename> <input_mode> [startVertex]
```

Where:
- `input_mode` can be either `adjacencyMatrix` or `adjacencyList`
- `startVertex` is optional and defaults to 1 (used for BFS/DFS algorithms)

### Examples

```bash
# Using adjacency matrix representation
./graph example.txt adjacencyMatrix 1

# Using adjacency list representation  
./graph example.txt adjacencyList 1

# Using different start vertex
./graph example.txt adjacencyMatrix 3
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

### Output

The program generates a comprehensive analysis file named `<input_filename>_<input_mode>_info.txt` containing:

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

The project now consists of a single unified file:
- **`graph.cpp`** - Complete graph library containing all classes and algorithms

This unified approach makes the library easy to:
- Use as an API for testing with larger graphs
- Integrate into other projects
- Maintain and modify
- Compile without dependencies

