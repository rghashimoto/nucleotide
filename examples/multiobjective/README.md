# Example: Multi-Objective Knapsack Solver (NSGA-II)

This example demonstrates how to solve a multi-objective knapsack optimization problem using the Non-dominated Sorting Genetic Algorithm II (NSGA-II) inside the `nucleotide` framework. The goal is to optimize two conflicting objectives: maximizing total item value while minimizing total weight.

---

## 1. Genome Architecture
The genome uses **`BitGenome`** (`[]bool`) to represent item selections:
```go
genome := make(nucleotide.BitGenome, genomeSize)
```
- **Why this option is optimal**: Selection/inclusion problems are represented as flat boolean arrays where the $i$-th bit represents the presence (`true`) or absence (`false`) of the $i$-th item. A `BitGenome` offers maximum performance since bitwise/boolean mutations and single-point crossovers can be performed with minimal CPU and memory allocations, keeping the multi-objective sorting loop low-overhead.

---

## 2. Code Execution Steps
1. **Define Environmental Data**: Set up 20 items with conflicting parameters (high-value items are heavy, light items are of low value).
2. **Define Multi-Objective Fitness Function**: 
   - Computes both total value and total weight as a vector of two float64 elements: `[]float64{totalValue, totalWeight}`.
3. **Configure Objective Directions**: Sets up `ObjectiveDirections` to direct the engine:
   - Objective 0: Maximize (`nucleotide.Maximize`)
   - Objective 1: Minimize (`nucleotide.Minimize`)
4. **Evolve via NSGA-II Engine**: The engine automatically detects the multi-objective profile and employs non-dominated sorting.
5. **Extract and Output Pareto Frontier**: Retrieves Rank 0 (non-dominated) solutions and saves the best-compromise solution genome.

---

## 3. Design Decisions Rationale

### Vector-Based Multi-Objective Optimization vs. Scalarized Weighted Sums
In many optimization tasks, multiple goals compete. A common approach is to combine these into a single scalar fitness using weights:
$$Fitness = w_1 \cdot \text{Value} - w_2 \cdot \text{Weight}$$
However, this approach has limitations:
1. **Scaling Bias**: It requires manual, arbitrary scaling factors to normalize different units (e.g., dollars vs. kilograms).
2. **Non-Convexity**: It cannot discover optimal solutions located in non-convex regions of the Pareto frontier.
3. **Single Point Output**: It yields a single compromise solution rather than a set of trade-off options.

By returning `[]float64{totalValue, totalWeight}` and using NSGA-II, the engine preserves the dimensional separation. It evaluates trade-offs naturally without pre-assigning relative importance to objectives.

### NSGA-II Sorting Mechanics
The engine utilizes two mechanisms to drive evolution towards the Pareto frontier:
1. **Non-Dominated Sorting**: The population is sorted into hierarchical fronts. An individual $A$ dominates $B$ if it is no worse than $B$ in all objectives and strictly better in at least one. Front 0 (Rank 0) contains the completely non-dominated individuals (the Pareto Frontier).
2. **Crowding Distance (Density Measure)**: Within each rank front, individuals are assigned a crowding distance measuring the local density in objective space. Individuals that are isolated receive a larger distance score.

### Selection via Crowding Comparison ($\prec_c$)
The **`GenericTournamentSelector`** leverages the crowding comparison operator during tournament selection. When comparing two individuals:
- The individual with the lower (better) non-dominated rank is preferred.
- If both reside on the same front (equal rank), the individual with the larger crowding distance is preferred.
This prevents the population from collapsing onto a single point of the trade-off space, ensuring that the final population is distributed evenly across the entire Pareto frontier.
