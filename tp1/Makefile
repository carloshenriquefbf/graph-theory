CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2
TARGET = performance_test
SOURCES = performance_test.cpp

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES)

clean:
	rm -f $(TARGET) *.csv

test: $(TARGET)
	./$(TARGET) results.csv

.PHONY: clean test
