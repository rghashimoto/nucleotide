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

> [!NOTE]
> **Loci** (pronounced *lo-sigh*) is the plural form of **Locus**. Throughout the library and documentation, we use **Locus** when referring to a single slot, and **Loci** when referring to multiple slots.

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

## Advanced Selection & Evolution Operators

Nucleotide provides a highly customizable operator execution model and advanced selection algorithms to prevent premature convergence and control selection pressure.

### 1. Multi-Operator Slices & Fallback Defaults
Instead of defining a single crossover or mutation strategy, `EngineConfig` supports multiple strategies in slices:
- **`Crossoverers` & `Mutators`**: Slices of strategies that run interchangeably. If empty, Nucleotide supplies smart defaults (`DefaultCrossoverer` and `DefaultMutator`) that automatically adjust at runtime to the type of genome being processed (`BitGenome`, `FloatGenome`, or `CategoricalGenome`).
- **Pondered Weights**: Use `CrossovererWeights` and `MutatorWeights` to configure custom probability distributions for each operator. If weights are omitted, the engine defaults to a smooth, alternating **Round-Robin** sequencing strategy.

### 2. Individual Lifetime (Age) Tracking
Individuals track their survival generation span using an `Age` property, initialized to `0` and automatically incremented at the end of each generation loop in the evolutionary engine. This age metric is used to model life expectancy and introduce biological selection penalties.

### 3. State-of-the-Art Selection Operators
All custom selectors implement the standard `Selector` interface:
* **`RouletteWheelSelector[E, S]`**: Proportional fitness selection with optional `AutoShift` capability to handle negative fitness boundaries.
* **`StochasticUniversalSamplingSelector[E, S]`**: Low-variance, zero-bias multi-pointer selection utilizing a single-spin buffer queue to ensure equal-interval selection across sequential calls.
* **`RankSelector[E, S]`**: Maps absolute fitness values to linear ranks ($1$ to $N$) with customizable `SelectionPressure` values. Prevents super-individuals from dominating early generations.
* **`BoltzmannSelector[E, S]`**: Standard temperature-scaled selection ($e^{f(x) / T}$) allowing exploration/exploitation weighting across epochs.

### 4. Advanced `GenericTournamentSelector[E, S]` Features
The built-in tournament selector can be enhanced using several advanced, fully opt-in dynamics:
- **Adaptive Diversity**: Sizing adapts dynamically depending on population standard deviation (reducing tournament size when diversity is low to encourage exploration).
- **Age Bias**: A custom penalty applied to competitor fitness proportional to their survival age (`adjustedFit = adjustedFit - age * AgeBias`) to prevent stagnation.
- **Hall of Fame competitor mixing**: Integrates historical elite individuals into active tournaments with a specified `HallOfFameProbability` to encourage competition.
- **Self-Adaptive Sizing**: Individuals can adaptively define their preferred tournament sizing (`TournamentSize`) through parameter genes or custom state interfaces implementing `SelfAdaptiveIndividual`.

## Multi-Objective Optimization (NSGA-II)

Nucleotide supports state-of-the-art **Multi-Objective Optimization** using the **NSGA-II (Nondominated Sorting Genetic Algorithm II)** algorithm out-of-the-box. This is ideal when you need to optimize conflicting metrics in tension, such as maximizing performance/throughput while minimizing cost/power consumption.

### Key NSGA-II Features:
- **Unified Fitness Signature**: Fitness is represented as `[]float64` to transparently scale from single-objective (length 1) to multi-objective environments.
- **Configurable Directions**: Define whether to `Maximize` or `Minimize` each objective independently via `ObjectiveDirections []ObjectiveDirection`.
- **Fast Non-Dominated Sorting**: Classifies individuals into sequential Pareto frontiers ($F_1, F_2, \dots$) based on mathematical dominance.
- **Crowding Distance & Density Calculation**: Scans boundary points and computes sparsity metrics to favor diverse, well-distributed solutions across the Pareto frontier.
- **Crowded Comparison Operator Selection**: The `GenericTournamentSelector` automatically applies rank-based and crowding-distance-based tournament selection when running in multi-objective mode.
- **Pareto Frontier Access**: Retrieve the best non-dominated solutions of a finished run via `engine.ParetoFrontier()`.

### Code Example:
```go
config := nucleotide.EngineConfig[MyEnv, MyState]{
    PopulationSize: 100,
    MaxGenerations: 50,
    FitnessFunc: func(g nucleotide.Genome, env MyEnv) []float64 {
        // Return multiple fitness scores
        return []float64{performance, cost}
    },
    Selector: nucleotide.GenericTournamentSelector[MyEnv, MyState]{Size: 3},
    
    // Objective 0: Maximize performance
    // Objective 1: Minimize cost
    ObjectiveDirections: []nucleotide.ObjectiveDirection{
        nucleotide.Maximize,
        nucleotide.Minimize,
    },
}

engine, _ := nucleotide.NewEngine(config)
engine.Run(def)

// Retrieve the optimal trade-offs (Rank 0 non-dominated solutions)
paretoFrontier := engine.ParetoFrontier()
```

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

### 4. Multi-Objective Optimization (NSGA-II)
Solve complex problems where multiple conflicting objectives must be optimized simultaneously (e.g. maximizing value while minimizing weight).
- **Example**: [multiobjective/main.go](file:///C:/Users/rafae/Desktop/golang/nucleotide/examples/multiobjective/main.go)
- **Goal**: Find the non-dominated Pareto Frontier trade-offs for a dual-objective Knapsack problem.

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
