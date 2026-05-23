package main

import (
	"context"
	"fmt"

	"github.com/rghashimoto/nucleotide"
)

type DummyEnv struct{}

func main() {
	// 1. Define the genome architecture
	def := nucleotide.NewDefinition[DummyEnv, struct{}]()

	color := def.AddLocus("Color", nucleotide.LocusBehavioral)
	color.AddGene("Red", func(ctx nucleotide.Context[DummyEnv, struct{}]) { fmt.Print("Color:Red ") })
	color.AddGene("Green", func(ctx nucleotide.Context[DummyEnv, struct{}]) { fmt.Print("Color:Green ") })
	color.AddGene("Blue", func(ctx nucleotide.Context[DummyEnv, struct{}]) { fmt.Print("Color:Blue ") })

	size := def.AddLocus("Size", nucleotide.LocusBehavioral)
	size.AddGene("Small", func(ctx nucleotide.Context[DummyEnv, struct{}]) { fmt.Print("Size:Small ") })
	size.AddGene("Medium", func(ctx nucleotide.Context[DummyEnv, struct{}]) { fmt.Print("Size:Medium ") })
	size.AddGene("Large", func(ctx nucleotide.Context[DummyEnv, struct{}]) { fmt.Print("Size:Large ") })

	material := def.AddLocus("Material", nucleotide.LocusBehavioral)
	material.AddGene("Wood", func(ctx nucleotide.Context[DummyEnv, struct{}]) { fmt.Print("Material:Wood ") })
	material.AddGene("Metal", func(ctx nucleotide.Context[DummyEnv, struct{}]) { fmt.Print("Material:Metal ") })
	material.AddGene("Plastic", func(ctx nucleotide.Context[DummyEnv, struct{}]) { fmt.Print("Material:Plastic ") })

	// 2. Define fitness function
	fitnessFunc := func(g nucleotide.Genome, env DummyEnv) []float64 {
		cg := g.(*nucleotide.CategoricalGenome[DummyEnv, struct{}])
		score := 0.0
		if cg.GeneIndices[1] == 0 { score += 1.0 }
		if cg.GeneIndices[2] == 2 { score += 1.0 }
		if cg.GeneIndices[3] == 1 { score += 1.0 }
		return []float64{score}
	}

	// 3. Initialize engine with automated population creation
	config := nucleotide.EngineConfig[DummyEnv, struct{}]{
		PopulationSize: 20,
		MaxGenerations: 20,
		FitnessFunc:    fitnessFunc,
		Selector:       nucleotide.GenericTournamentSelector[DummyEnv, struct{}]{Size: 3},
		Crossoverers:   []nucleotide.WeightedCrossoverer{{Crossoverer: nucleotide.SinglePointCrossover{}}},
		Mutators:       []nucleotide.WeightedMutator{{Mutator: nucleotide.CategoricalMutator{Probability: 0.1}}},
		Elitism:        1,
		Env:            DummyEnv{},
	}
	engine, err := nucleotide.NewEngine[DummyEnv, struct{}](config)
	if err != nil {
		panic(err)
	}

	// 4. Run evolution
	best, _ := engine.Run(def)

	fmt.Printf("\nEvolution finished!\n")
	fmt.Printf("Best individual fitness: %.1f/3.0\n", best.Fitness[0])
	fmt.Print("Expressed features: ")
	best.Express(context.Background(), DummyEnv{})
	fmt.Println()
}
