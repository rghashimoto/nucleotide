# Example: Concurrent Multi-Island Climate Solver

This example demonstrates how to solve a co-dependent parameter and route optimization problem using a parallelized **`MultiIslandEngine`** inside the `nucleotide` framework. The goal is to evolve a generalist drone delivery profile capable of surviving under three heterogeneous climates (Sunny, High Winds, and Monsoon) by isolating populations on separate islands and periodically migrating elite individuals.

---

## 1. Genome Architecture
The genome is represented using a **`CompositeGenome`** grouping categorical parameters (Battery Capacity, Cruise Speed) and sequence permutations (Waypoints Route).

```go
def := nucleotide.NewDefinition[WeatherClimate, *DroneState]()
batteryLocus := def.AddLocus("BatteryCapacity", nucleotide.LocusParameter)
speedLocus := def.AddLocus("CruiseSpeed", nucleotide.LocusParameter)
routeLocus := def.AddLocus("WaypointsRoute", nucleotide.LocusSequence)
```

- **Why this option is optimal**: For co-evolutionary and distributed genetic algorithms, the Multi-Island Model addresses the premature convergence problem of single-population GAs. A single large population is vulnerable to a "super-individual" dominating the population, trapping the search in local optima. By partitioning the global population into independent sub-populations (islands) that explore different regions of the search space, diversity is preserved.

---

## 2. Code Execution Steps
1. **Define Environmental climates**: Configure three distinct `WeatherClimate` states representing Sunny & Calm, High Winds (increased drag), and Monsoon (reduced speed, minor drag) conditions.
2. **Configure Multi-Island Engine**:
   - Spawns 3 independent islands.
   - Allocates a local population of 20 individuals per island.
   - Defines an `EnvFactory` callback assigning a specific climate to each island:
     ```go
     envFactory := func(islandIndex int) WeatherClimate { return climates[islandIndex] }
     ```
3. **Set Migration Policies**:
   - Evolve locally for 5 generations (`MigrationInterval = 5`).
   - Move the top 2 individuals (`MigrationRate = 2`) via a ring path (`TopologyRing`).
   - Replace the weakest individuals in the destination island (`PolicyBestReplaceWorst`).
4. **Run Evolution**: Runs concurrent epoch loops utilizing Go routines.
5. **Output Evolved Generalist**: Prints the best genome capable of performing across all weather climates.

---

## 3. Design Decisions Rationale

### Preserving Global Diversity through Speciation (Sub-Population Isolation)
In standard genetic algorithms, selection pressure tends to reduce diversity as search converges. In the Multi-Island Model, each island acts as an isolated genetic repository:
- **Niche Exploration**: Because each island runs its own selection, crossover, and mutation loops, they search different regions of the fitness landscape.
- **Local Adaptation**: In this example, Island 1 adapts to low-energy calm conditions, Island 2 adapts to high-drag wind conditions (favoring large batteries), and Island 3 adapts to monsoon constraints.

### Ring Topology and Diffusion Rate
Migration allows islands to share discovered genetic advancements. The configuration parameters are chosen to balance exploration against premature homogenization:
- **`TopologyRing`**: Connects islands in a single directional loop (Island 1 $\rightarrow$ Island 2 $\rightarrow$ Island 3 $\rightarrow$ Island 1). This linear structure limits the rate of genetic diffusion. An elite individual discovered on Island 1 cannot instantly dominate Island 3, preserving Island 3's local progress for at least two migration intervals.
- **`PolicyBestReplaceWorst`**: Replaces the worst individuals in the target island with the best from the source. This ensures that beneficial mutations are introduced into other islands without destroying their local elites.

### Co-Evolution of Generalist Genotypes
Because individuals migrate between environments, they are subjected to different fitness landscapes over their lifespan. A genome that specializes exclusively in sunny conditions will have low fitness and face extinction when migrated to the High Winds or Monsoon islands. 
Consequently, only "generalist" genomes—those that select robust battery capacities and robust route sequence orderings that succeed under all three climates—can survive long-term across migrations. This drives co-evolutionary robustness.

### Concurrent Multi-Core Execution
The engine leverages Go's concurrency model:
- Each island's local evolution loop runs in its own goroutine.
- Generations within an epoch run concurrently across all available CPU cores.
- The engine uses a synchronization barrier (`sync.WaitGroup`) at the end of each migration interval to halt evolution, perform cross-island migrations safely in a single thread without race conditions, and then resume concurrent execution. This minimizes wall-clock runtimes.
