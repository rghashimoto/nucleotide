# Example: OneMax Binary Optimization

This example demonstrates how to solve the classic **OneMax** optimization problem using the `nucleotide` framework. The goal is to evolve a flat binary string to contain the maximum number of `true` values.

---

## 1. Genome Architecture
The genome is represented using **`BitGenome`** (`[]bool`):
```go
genome := make(nucleotide.BitGenome, genomeSize)
```
- **Why this option is optimal**: For flat binary optimization, representing states as a contiguous slice of booleans has zero memory map or key lookup overhead. It provides maximum CPU cache alignment and minimizes garbage collection overhead, making it far superior to using structured `CategoricalGenome` wrappers for simple binary decisions.

---

## 2. Code Execution Steps
1. **Initialize Population**: Generates a starting population of 100 individuals with random binary strings.
2. **Configure Engine**: 
   - Uses a single-objective fitness function that counts the number of `true` values in the `BitGenome`.
   - Sets the `AgeBiasedMutation` parameter to `true` to scale mutation rates for older individuals.
3. **Run Evolution**: Progresses through generations up to a limit of 40.
4. **Output Results**: Prints the best evolved genome string and its fitness value.

---

## 3. Design Decisions Rationale

### Fitness Function
The fitness function is simple and deterministic:
```go
fitnessFunc := func(g nucleotide.Genome, env EmptyEnv) []float64 {
    bg := g.(nucleotide.BitGenome)
    score := 0.0
    for _, bit := range bg {
        if bit {
            score += 1.0
        }
    }
    return []float64{score}
}
```
Evaluating the count of `true` values directly maps to the objective without complex transformations, keeping computation costs low.

### Selection Operator
We utilize **`GenericTournamentSelector`** with a tournament size of `3`:
- A tournament size of 3 balances selection pressure. Too small (e.g., 2) slows down convergence; too large (e.g., >8) leads to premature convergence by copying the same top individuals too quickly, destroying diversity.

### Mutation Operator
- **`BitFlipMutator`**: The natural mutation operator for binary strings, flipping individual booleans with a specified probability (`0.02`).
- **`AgeBiasedMutation`**: By enabling age-biased scaling, individuals that survive multiple generations have their mutation rates scaled up. This prevents local stagnation by forcing older, static individuals to mutate, encouraging exploration in the search space.

### Elitism
We set `Elitism = 1`. This guarantees that the best-performing binary string from the previous generation is carried over to the next generation without modification, preventing regression while allowing the rest of the population to explore.
