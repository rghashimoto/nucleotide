# Nucleotide

**Nucleotide** is a highly modular and type-safe Genetic Algorithm (GA) framework for Go, designed to evolve complex behaviors and parameters for software agents and simulations.

It introduces the concept of **Categorical Genomes**, where evolution doesn't just tune numbers, but selects specific behaviors (functions) and configurations that can be directly executed in a simulated or real-world environment.

## Key Features

- **Generic Environments (`[E any]`)**: Pass any type (Database connections, World simulations, API clients) directly to your genes.
- **Typed Loci Architecture**:
    - **Behavioral**: Genes that execute logic via callbacks.
    - **Parameter**: Genes that provide data values for behavioral genes.
    - **Configuration**: Internal framework settings that can themselves be evolved.
- **Context-Aware Expression**: Execute evolved individuals with `context.Context` for safe cancellation and timeouts.
- **Robust Serialization**: Save winning genomes to JSON using stable IDs. Load them in production with zero friction.
- **Modern Go**: Built with Go 1.22+ and fully leverages Generics.

## Core Concepts: Loci vs. Genes

To understand Nucleotide, it's essential to distinguish between a **Locus** and a **Gene**:

### The Biological Analogy
Imagine the trait for **Eye Color**. 
- The **Locus** is the specific position on a chromosome that determines eye color. Every human has this "slot".
- The **Gene** (or allele) is the specific version occupying that slot—such as Blue, Brown, or Green.

### The Software Analogy
In a software system, imagine a **Sorting Strategy**:
- The **Locus** is the abstract "Sorting" component in your architecture.
- The **Genes** are the concrete implementations you can plug in: `QuickSort`, `MergeSort`, or `HeapSort`.

Evolution in Nucleotide works by testing which **Gene** performs best at each **Locus** given a specific environment.

## Advanced Dynamics

### 1. Execution Order
Nucleotide allows the evolution of the **order in which genes are expressed**. In many systems, the sequence of operations is as critical as the operations themselves. For example, in a processing pipeline, executing `ValidateData` before `SaveToDB` is mandatory, but the order of optional `Enrichment` steps might yield different results depending on environmental constraints (like latency or data availability).

### 2. Parameterization & Adaptation
**Parameter Genes** allow a specific algorithm (Gene) to operate differently to best suit its environment. 
- **Example**: In a **K-Means Clustering** algorithm, the number of clusters (*k*) is a critical parameter. 
- A locus could define the "Cluster Count", and evolution would find the optimal *k* for the current dataset, allowing the same K-Means gene to adapt its behavior without changing its core logic.

## Installation

```bash
go get github.com/rghashimoto/nucleotide
```

## Usage Levels

### 1. Basic Usage (Optimization)
Ideal for solving classic optimization problems like OneMax or Knapsack using simple bitstring or float genomes.
- **Example**: [onemax/main.go](file:///C:/Users/rafae/Desktop/golang/nucleotide/examples/onemax/main.go)
- **Goal**: Find the bitstring with the maximum number of `true` values.

### 2. Structured Usage (Component Selection)
Use Categorical Genomes to select the best combination of components or traits for an entity.
- **Example**: [structured/main.go](file:///C:/Users/rafae/Desktop/golang/nucleotide/examples/structured/main.go)
- **Goal**: Evolve an object with the best combination of `Color`, `Size`, and `Material`.

### 3. Advanced Usage (Behavioral Evolution)
Evolve agents that interact with a dynamic environment. Genes are functions that consume resources or change the state of the world.
- **Example**: [advanced/main.go](file:///C:/Users/rafae/Desktop/golang/nucleotide/examples/advanced/main.go)
- **Goal**: Evolve a survival strategy (Glutton vs. Frugal) in a world with limited food.

## Customization & Extensibility

Nucleotide is built to be extended. You can customize almost every aspect of the genetic process by providing your own functions:

| Function Type | Purpose | Usage |
| :--- | :--- | :--- |
| `FitnessFunc[E]` | Defines how to score an individual. | `config.FitnessFunc = myFunc` |
| `ElitismFunc[E]` | Defines which individuals survive to the next generation. | `config.ElitismFunc = nucleotide.TopNElitism` |
| `PopulationFunc[E]` | Defines how the initial population is created. | `config.PopulationFunc = myPopFactory` |
| `Sequencer[E]` | Controls the order in which behavioral genes are expressed. | Add to "Execution Order" Locus |
| **Gene Callbacks** | The core logic executed when a gene is expressed. | `locus.AddGene("ID", myCallback)` |

### Example: Custom Sequencer
You can control the execution flow based on the selected genes:
```go
execLocus.AddConfigGene("MyOrder", func(ctx SequencingContext[E]) []int {
    // Return a custom slice of indices to define the execution order
    return []int{1, 0, 2} 
})
```

---

## From Evolution to Production

Nucleotide is designed to bridge the gap between AI research and production deployment.

1. **Evolve**: Run the Genetic Engine to find the best individual.
2. **Save**: Export the genome to a portable JSON file.
   ```go
   best.Save("production_config.json")
   ```
3. **Deploy**: Load the JSON in your production service and execute it.
   ```go
   loadedGenome, _ := nucleotide.LoadGenome(prodDef, "production_config.json")
   agent := nucleotide.NewIndividual(loadedGenome)
   agent.Express(ctx, liveEnvironment)
   ```

## License
BSD-3-Clause
