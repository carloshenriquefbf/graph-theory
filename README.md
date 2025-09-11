# Graph theory - COS242

Usage:

To compile the program, use the following command:
```
g++ -o graph graph.cpp
```

The command line arguments are:

```
<input_filename> <input_mode>
```

Where `input_mode` can be either `adjacencyMatrix` or `adjacencyList`, e.g.:

```bash
./graph $(echo "example.txt adjacencyMatrix")
```

Where `example.txt` is a text file containing the graph data. It should be in the following format:

```
5
1 2
2 5
5 3
4 5
1 5
```

The first line indicates the number of vertices in the graph. Each subsequent line represents an edge between two vertices.

The program will output the graph statistics in a file named `<input_filename>_<input_mode>_info.txt` in the following format:

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

