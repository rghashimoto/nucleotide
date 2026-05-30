# Example: Behavioral Survival Agent

This example demonstrates how to build and evolve agents with behavioral phenotypes using the `nucleotide` framework. The agent interacts with a mutable environment (`World`) and updates its own state (`AgentState`) using sequences of functional genes.

---

## 1. Genome Architecture
The agent is defined with two loci using a **`CategoricalGenome`**:
```go
def := nucleotide.NewDefinition[*World, AgentState]()
gather := def.AddLocus("Gather", nucleotide.LocusBehavioral)
metabolism := def.AddLocus("Metabolism", nucleotide.LocusBehavioral)
```
- **Why this option is optimal**: For behavioral simulation, expressing genes as direct execution callbacks is far more robust than interpreting numeric genomes at runtime. By binding anonymous Go functions to genes, the genetic structure directly reflects the behavior pipeline. The framework manages execution, removing the need for a separate parser or state machine interpreter.

---

## 2. Code Execution Steps
1. **Define Environmental and Agent States**: 
   - `World` models global resources (e.g., `FoodAvailable`).
   - `AgentState` models individual-level telemetry (e.g., `Energy`).
2. **Define Loci and Behavioral Genes**: Registers `Gather` and `Metabolism` loci. Each locus features alternative gene implementations modifying both `World` and `AgentState`.
3. **Configure and Run Evolution**: Evolves a population of 20 individuals over 5 generations to maximize energy and favor frugal gathering.
4. **Save Evolved Genome**: Serializes the best-performing individual's genome structure to `best_genome.json`.
5. **Simulate Production Hot-Load**: Recreates a fresh `Definition` in a mock production process, loads the serialized genome, and executes its behavior against a new world state.

---

## 3. Design Decisions Rationale

### Mutable State Isolation and Decoupling
To prevent race conditions and cross-talk during concurrent evaluation, state is split into two structures:
- **Global Environment (`World`)**: Shared resource state. Inside the fitness function, a local copy of the world is instantiated for each evaluation to maintain isolation:
  ```go
  localEnv := &World{FoodAvailable: env.FoodAvailable}
  ```
- **Agent State (`AgentState`)**: Encapsulates private variables (energy) belonging to a single individual:
  ```go
  ind := nucleotide.NewIndividual[*World, AgentState](cg)
  ind.State = AgentState{Energy: 0}
  ```
This separation allows the engine to run evaluations in parallel safely since no two individuals share memory.

### Locus Execution Order
The order in which loci are registered to the `Definition` determines their execution sequence:
1. `Gather` runs first: Agent attempts to gather food and convert it to energy.
2. `Metabolism` runs second: Agent digests its current energy.
If the order were reversed, an agent starting with zero energy would fail to metabolize, gather food, and end the step with excess energy. Registration-based ordering ensures a predictable, deterministic execution pipeline without requiring manual scheduling code.

### Robust JSON Serialization
Loaded genomes map string identifiers rather than raw array indices:
- `Save` serializes the names of active genes alongside their locus names.
- `LoadGenome` matches these string names back to the functions registered in the provided `Definition`.
This decoupling makes saved genomes resilient to structural modifications in the code. If a new gene is added in the middle of a locus definition, or if registration order is modified, the JSON file still loads correctly because mapping is resolved by name, not array offsets.
