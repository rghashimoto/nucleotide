package main

import (
	"fmt"
	"math/rand"

	"github.com/rghashimoto/nucleotide"
)

type KnapsackEnv struct {
	ItemValues  []float64
	ItemWeights []float64
}

func main() {
	popSize := 50
	genomeSize := 20
	maxGens := 40

	// Set up environment with 20 items of conflicting value and weight
	// Some items are highly valuable but very heavy, others are light but less valuable
	env := KnapsackEnv{
		ItemValues:  make([]float64, genomeSize),
		ItemWeights: make([]float64, genomeSize),
	}
	for i := 0; i < genomeSize; i++ {
		env.ItemValues[i] = float64(10 + i*2)            // Values range from 10 to 48
		env.ItemWeights[i] = float64(5 + (genomeSize-i)*3) // Weights range from 5 to 62 (conflicting)
	}

	fitnessFunc := func(g nucleotide.Genome, env KnapsackEnv) []float64 {
		bg := g.(nucleotide.BitGenome)
		totalValue := 0.0
		totalWeight := 0.0
		for i, selected := range bg {
			if selected {
				totalValue += env.ItemValues[i]
				totalWeight += env.ItemWeights[i]
			}
		}
		// Objective 0: Maximize total value
		// Objective 1: Minimize total weight
		return []float64{totalValue, totalWeight}
	}

	initialPop := make(nucleotide.Population[KnapsackEnv, struct{}], popSize)
	for i := 0; i < popSize; i++ {
		genome := make(nucleotide.BitGenome, genomeSize)
		for j := 0; j < genomeSize; j++ {
			genome[j] = rand.Float64() < 0.3 // start sparse
		}
		initialPop[i] = nucleotide.NewIndividual[KnapsackEnv, struct{}](genome)
	}

	config := nucleotide.EngineConfig[KnapsackEnv, struct{}]{
		PopulationSize: popSize,
		MaxGenerations: maxGens,
		FitnessFunc:    fitnessFunc,
		Selector:       nucleotide.GenericTournamentSelector[KnapsackEnv, struct{}]{Size: 4},
		Crossoverers:   []nucleotide.WeightedCrossoverer{{Crossoverer: nucleotide.SinglePointCrossover{}}},
		Mutators:       []nucleotide.WeightedMutator{{Mutator: nucleotide.BitFlipMutator{Probability: 0.05}}},
		Elitism:        2,
		Env:            env,
		// Define optimization directions for each objective:
		// Objective 0 (Value) -> Maximize
		// Objective 1 (Weight) -> Minimize
		ObjectiveDirections: []nucleotide.ObjectiveDirection{
			nucleotide.Maximize,
			nucleotide.Minimize,
		},
	}

	engine, err := nucleotide.NewEngine[KnapsackEnv, struct{}](config)
	if err != nil {
		panic(err)
	}
	engine.Population = initialPop

	fmt.Printf("Starting Multi-Objective Optimization (NSGA-II)...\n")
	_, err = engine.Run(nil)
	if err != nil {
		panic(err)
	}

	fmt.Printf("\nEvolution finished!\n")

	// Get Pareto frontier (non-dominated solutions, Rank 0)
	pareto := engine.ParetoFrontier()
	fmt.Printf("Found %d non-dominated Pareto-optimal solutions:\n", len(pareto))
	fmt.Printf("%-5s | %-12s | %-12s | %-30s\n", "Index", "Total Value", "Total Weight", "Genome representation")
	fmt.Println("--------------------------------------------------------------------------------")
	for i, ind := range pareto {
		bg := ind.Genome.(nucleotide.BitGenome)
		genesStr := ""
		for _, b := range bg {
			if b {
				genesStr += "1"
			} else {
				genesStr += "0"
			}
		}
		fmt.Printf("%-5d | %-12.1f | %-12.1f | %-30s\n", i+1, ind.Fitness[0], ind.Fitness[1], genesStr)
	}
}
