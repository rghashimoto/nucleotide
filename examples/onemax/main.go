package main

import (
	"fmt"
	"math/rand"

	"github.com/rghashimoto/nucleotide"
)

type EmptyEnv struct{}

func main() {
	popSize := 100
	genomeSize := 50
	maxGens := 50

	fitnessFunc := func(g nucleotide.Genome, env EmptyEnv) []float64 {
		bg := g.(nucleotide.BitGenome)
		count := 0
		for _, b := range bg {
			if b {
				count++
			}
		}
		return []float64{float64(count)}
	}

	initialPop := make(nucleotide.Population[EmptyEnv, struct{}], popSize)
	for i := 0; i < popSize; i++ {
		genome := make(nucleotide.BitGenome, genomeSize)
		for j := 0; j < genomeSize; j++ {
			genome[j] = rand.Float64() < 0.5
		}
		initialPop[i] = nucleotide.NewIndividual[EmptyEnv, struct{}](genome)
	}

	config := nucleotide.EngineConfig[EmptyEnv, struct{}]{
		PopulationSize: popSize,
		MaxGenerations: maxGens,
		FitnessFunc:    fitnessFunc,
		Selector:       nucleotide.GenericTournamentSelector[EmptyEnv, struct{}]{Size: 3},
		Crossoverers:   []nucleotide.WeightedCrossoverer{{Crossoverer: nucleotide.SinglePointCrossover{}}},
		Mutators:       []nucleotide.WeightedMutator{{Mutator: nucleotide.BitFlipMutator{Probability: 0.01}}},
		Elitism:        1,
		Env:            EmptyEnv{},
	}
	engine, err := nucleotide.NewEngine[EmptyEnv, struct{}](config)
	if err != nil {
		panic(err)
	}
	engine.Population = initialPop

	best, _ := engine.Run(nil)

	fmt.Printf("\nEvolution finished!\n")
	fmt.Printf("Best individual fitness: %.2f/%d\n", best.Fitness[0], genomeSize)
}
