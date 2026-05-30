# Example: Multi-Chromosomal Logistics Solver

This example demonstrates how to solve a co-dependent parameter and routing optimization problem using a **`CompositeGenome`** inside the `nucleotide` framework. The goal is to optimize both a drone's physical configurations (Battery Capacity, Cruise Speed, and Startup Actions) and its visiting sequence for multiple delivery locations (a Travelling Salesperson Problem).

---

## 1. Genome Architecture
The genome is represented using a **`CompositeGenome`**, which acts as a multi-chromosomal container grouping distinct types of chromosomes:
- **`categorical` Chromosome**: Manages discrete selections (`LocusBehavioral` and `LocusParameter` genes for drone startup actions, battery types, and speed levels).
- **`DeliveryRoute` Chromosome**: A sequence chromosome (`LocusSequence`) managing a unique permutation of customer indices $[1, 5]$.

```go
def := nucleotide.NewDefinition[LogisticsEnv, *LogisticsState]()
actionLocus := def.AddLocus("StartupAction", nucleotide.LocusBehavioral)
capacityLocus := def.AddLocus("BatteryCapacity", nucleotide.LocusParameter)
speedLocus := def.AddLocus("CruiseSpeed", nucleotide.LocusParameter)
routeLocus := def.AddLocus("DeliveryRoute", nucleotide.LocusSequence)
```

- **Why this option is optimal**: Real-world logistics tasks require optimizing both parameters (e.g., speed, battery sizes) and orderings (e.g., the delivery path). If represented in a single flat array, normal genetic operators fail: single-point crossover on the route would generate duplicate customer visits or omit deliveries, while sequence-based operators (like PMX) would corrupt categorical configurations. A `CompositeGenome` decouples these chromosomes, applying standard operators to parameters and specialized permutation-preserving operators to sequences.

---

## 2. Code Execution Steps
1. **Define Environmental and State Data**:
   - `LogisticsEnv` holds customer distances, package values, and delivery time limits.
   - `LogisticsState` records startup actions, total time spent, and total priority value delivered.
2. **Define Multi-Chromosomal Definition**: Registers three discrete parameter/behavioral loci and one sequence locus.
3. **Fitness Evaluation**: Extract the active battery/speed parameters and delivery route sequence from the `CompositeGenome`, simulate the flight profile under physics constraints (energy depletion and time expiration), and return a scalar score:
   $$\text{Fitness} = \text{DeliveredValue} - (\text{TimeSpent} \times 5.0)$$
4. **Evolve population**: Evolves 50 individuals over 60 generations.
5. **Phenotypic Expression and Serialization**: Executes startup actions, prints the optimal route sequence, saves the composite genome to `best_dispatch_plan.json`, and loads it back in a validation step.

---

## 3. Design Decisions Rationale

### Specialized Operator Isolation
Standard crossover and mutation operators do not preserve permutation uniqueness constraints. If a simple slice swap happens on a route sequence $[1, 2, 3, 4, 5]$, it easily yields $[1, 2, 2, 4, 5]$, violating the requirement that each customer is visited exactly once. 
To resolve this:
- **`PMXCrossover` (Partially Mapped Crossover)**: Transports sequence segments from parents to offspring, mapping values between parents to resolve duplicates, ensuring a valid permutation.
- **`SwapMutator`**: Mutates sequences by swapping two positions, maintaining index uniqueness.
The `nucleotide` framework handles this mapping internally. It automatically applies standard `SinglePointCrossover` and `CategoricalMutator` to the `categorical` sub-chromosome, while applying `PMXCrossover` and `SwapMutator` specifically to the `DeliveryRoute` chromosome.

### Parameter Extraction and API Type Safety
Parameters and sequences are extracted through helper methods on `Individual`:
```go
bestRoute := bestInd.GetSequence("DeliveryRoute")
bestBattery := bestInd.GetParameter("BatteryCapacity")
bestSpeed := bestInd.GetParameter("CruiseSpeed")
```
This isolates the downstream evaluation and simulation code from the underlying map representation of `CompositeGenome`, allowing developers to read genetic variables using standard Go types.

### Schema-Driven Serialization
Composite genomes contain highly complex structures (nested maps, slices, and interfaces). Standard JSON serialization loses type information. The `nucleotide` framework resolves this through schema-driven loading:
- `SaveGenome` writes the raw values and string-mapped index records.
- `LoadGenome` passes the JSON data along with the schema `Definition`:
  ```go
  loadedGenome, err := nucleotide.LoadGenome(def, filename)
  ```
The loader reads the active gene IDs and matches them back to the registered behavioral functions and parameter values defined in `def`, reconstructing a typed composite genome structure.
