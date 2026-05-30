# Example: Structured Component Selection

This example demonstrates how to solve a discrete component selection problem (Color, Size, Material) using a **`CategoricalGenome`** inside the `nucleotide` framework. The goal is to evolve an individual with specific categorical traits that maximize a predefined compatibility score.

---

## 1. Genome Architecture
The genome is represented using **`CategoricalGenome`**:
```go
def := nucleotide.NewDefinition[DummyEnv, struct{}]()
color := def.AddLocus("Color", nucleotide.LocusBehavioral)
color.AddGene("Red", func(ctx nucleotide.Context[DummyEnv, struct{}]) { ... })
// ...
```
- **Why this option is optimal**: When optimizing configurations with discrete, mutually exclusive options (e.g., specific colors or sizes), using flat binary strings (`BitGenome`) or raw integer strings (`[]int`) is prone to generating out-of-bounds representations during mutation or crossover. A `CategoricalGenome` paired with a `Definition` enforces architectural constraints. It ensures that each locus (Color, Size, Material) only ever holds a valid gene index within the bounds of its defined options, completely eliminating the need for post-mutation validation or penalty functions.

---

## 2. Code Execution Steps
1. **Define Genome Structure**: Creates a `Definition` containing three distinct loci (Color, Size, Material), each populated with three concrete genes representing behavior callbacks.
2. **Initialize Engine**: Sets up a `NewEngine` with a population of 20 individuals and a generation limit of 20.
3. **Configure Operators**: 
   - Uses `SinglePointCrossover` to exchange whole loci traits between parents.
   - Uses `CategoricalMutator` with a `0.1` probability to select alternative genes within the valid range for a locus.
4. **Evaluate and Evolve**: Evaluates the fitness function over generations, carrying over the best individual via `Elitism = 1`.
5. **Express Phenotype**: Invokes `best.Express(...)` to run the behavior callbacks for the optimal traits.

---

## 3. Design Decisions Rationale

### Fitness Function
The fitness function scores individuals based on specific target indices in the categorical genome:
```go
fitnessFunc := func(g nucleotide.Genome, env DummyEnv) []float64 {
    cg := g.(*nucleotide.CategoricalGenome[DummyEnv, struct{}])
    score := 0.0
    if cg.GeneIndices[1] == 0 { score += 1.0 } // Target Color: Red
    if cg.GeneIndices[2] == 2 { score += 1.0 } // Target Size: Large
    if cg.GeneIndices[3] == 1 { score += 1.0 } // Target Material: Metal
    return []float64{score}
}
```
Representing chromosomes as direct offsets (`GeneIndices`) allows for low-overhead index checking, avoiding string comparison costs during the evaluation loop.

### Selection Operator
We utilize **`GenericTournamentSelector`** with a size of `3`:
- Standardizes selection pressure. In a small search space of 27 possible combinations ($3 \times 3 \times 3$), a tournament size of 3 provides enough selection pressure to rapidly isolate target genes while still allowing minor variants to mutate and prevent premature loss of diversity.

### Mutation and Crossover Operators
- **`CategoricalMutator`**: Unlike numeric mutators that add random noise, this operator replaces a locus's gene index with a different index chosen uniformly from the valid range of genes defined for that specific locus. This guarantees the mutation results in a valid phenotype.
- **`SinglePointCrossover`**: Splitting parent slices along locus boundaries ensures that offspring inherit coherent traits (e.g., a complete "Size" gene from one parent and a complete "Material" gene from another) rather than producing corrupted or malformed gene indices.

### Phenotypic Expression
- By designating loci as `LocusBehavioral`, the framework associates functional Go callbacks with each gene. Calling `best.Express(context.Background(), DummyEnv{})` sequentially executes the callbacks associated with the active genes in the genome. This decouples genetic search logic from behavioral execution.
