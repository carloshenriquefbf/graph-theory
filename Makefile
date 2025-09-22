CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2
TARGET = graph
SOURCES = graph.cpp
HEADERS = 

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES)

clean:
	rm -f $(TARGET) *.txt

test: $(TARGET)
	./$(TARGET) example.txt adjacencyMatrix 1
	./$(TARGET) example.txt adjacencyList 1

.PHONY: clean test
