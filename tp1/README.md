# Estudos de Caso - Performance de Grafos

Este projeto implementa testes de performance para comparar diferentes representações de grafos (matriz de adjacência vs lista de adjacência) utilizando a biblioteca de grafos desenvolvida.

## Arquivos do Projeto

### Programas de Teste
- **`performance_test.cpp`** - Programa completo de testes (para grafos grandes)
- **`quick_test.cpp`** - Programa simplificado para testes rápidos
- **`Makefile`** - Compilação dos programas

### Dados de Teste
- **`grafo_1.txt`** - Grafo com 10.000 vértices
- **`grafo_2.txt`** - Grafo com 49.948 vértices  
- **`test_small.txt`** - Grafo pequeno para testes rápidos (100 vértices)

### Análise
- **`analysis.ipynb`** - Notebook Jupyter para análise dos resultados
- **`quick_results.csv`** - Resultados dos testes (formato CSV)

## Como Executar

### 1. Compilação
```bash
# Compilar programa completo
make

# Ou compilar programa rápido
g++ -std=c++17 -Wall -Wextra -O2 -o quick_test quick_test.cpp
```

### 2. Execução dos Testes
```bash
# Teste rápido (recomendado para demonstração)
./quick_test quick_results.csv

# Teste completo (pode demorar muito para grafos grandes)
./performance_test results.csv
```

### 3. Análise dos Resultados
```bash
# Abrir o notebook Jupyter
jupyter notebook analysis.ipynb
```

## Estudos de Caso Implementados

### 1. Comparação de Memória
- Mede o uso de memória (MB) para cada representação
- Calcula diferenças percentuais
- Analisa trade-offs de espaço

### 2. Comparação de Tempo de Execução
- **BFS**: 100 buscas em largura com vértices aleatórios
- **DFS**: 100 buscas em profundidade com vértices aleatórios
- Mede tempo médio por busca (ignorando I/O)

### 3. Análise de Árvores de Busca
- Determina pais dos vértices 10, 20, 30
- Testa com pontos de partida 1, 2, 3
- Compara árvores BFS vs DFS

### 4. Cálculo de Distâncias
- Distância entre pares: (10,20), (10,30), (20,30)
- Usa BFS como primitiva
- Verifica consistência entre representações

### 5. Componentes Conexas
- Conta número de componentes
- Identifica maior e menor componente
- Lista vértices por componente (ordenado por tamanho)

### 6. Diâmetro do Grafo
- **Exato**: Calcula distância entre todos os pares
- **Aproximado**: Algoritmo amostral para grafos grandes
- Compara precisão do algoritmo aproximado

## Resultados Esperados

### Memória
- **Lista de Adjacência**: O(n + m) - mais eficiente para grafos esparsos
- **Matriz de Adjacência**: O(n²) - mais eficiente para grafos densos

### Performance
- **Lista de Adjacência**: Melhor para grafos esparsos
- **Matriz de Adjacência**: Melhor para grafos densos
- **BFS/DFS**: Tempo O(n + m) em ambas as representações

### Consistência
- Algoritmos devem produzir resultados idênticos
- Diferenças apenas em performance, não em correção

## Bibliotecas Utilizadas

O projeto utiliza a biblioteca de grafos unificada (`graph.cpp`) que contém:

- **Representações**: AdjacencyListGraph, AdjacencyMatrixGraph
- **Algoritmos**: BFS, DFS, Distance, Diameter, ConnectedComponents
- **Princípios OOP**: Herança, reutilização de código, templates

## Comentários de Reutilização

O código está marcado com comentários `// REUTILIZAÇÃO:` em todos os pontos onde há reutilização de código, demonstrando os princípios de herança e DRY (Don't Repeat Yourself).

## Exemplo de Uso

```cpp
// Carregar grafo
AdjacencyListGraph graph = GraphFileReader::readFromFile<AdjacencyListGraph>("grafo.txt");

// Executar BFS
BFSAlgorithm<AdjacencyListGraph> bfs(&graph);
bfs.execute(1);

// Obter resultados
int parent = bfs.getParent(10);
int level = bfs.getLevel(10);
```

## Análise com Jupyter

O notebook `analysis.ipynb` fornece:

- Gráficos comparativos de performance
- Análise estatística dos resultados
- Tabelas de resumo
- Conclusões e recomendações
- Visualizações interativas

Execute o notebook para obter uma análise completa dos resultados dos testes.
