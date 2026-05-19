package main

import (
	"context"
	"fmt"
	"time"

	"github.com/rghashimoto/nucleotide"
)

// World is the simulated environment.
type World struct {
	FoodAvailable int
}

func main() {
	// 1. Define the genome architecture for evolution
	def := nucleotide.NewDefinition[*World]()

	behavior := def.AddLocus("Behavior", nucleotide.LocusBehavioral)
	behavior.AddGene("Glutton", func(ctx nucleotide.Context[*World]) {
		if ctx.Env.FoodAvailable > 0 {
			ctx.Env.FoodAvailable -= 2
		}
	})
	behavior.AddGene("Frugal", func(ctx nucleotide.Context[*World]) {
		if ctx.Env.FoodAvailable > 0 {
			ctx.Env.FoodAvailable -= 1
		}
	})

	// 2. Define fitness function
	fitnessFunc := func(g nucleotide.Genome, env *World) float64 {
		cg := g.(*nucleotide.CategoricalGenome[*World])
		localEnv := &World{FoodAvailable: env.FoodAvailable}
		ind := nucleotide.NewIndividual[*World](cg)
		ind.Express(context.Background(), localEnv)
		
		score := 0.0
		if localEnv.FoodAvailable >= 0 {
			score = float64(localEnv.FoodAvailable)
		}
		if cg.GeneIndices[1] == 1 { score += 10.0 }
		return score
	}

	// 3. Initialize engine
	config := nucleotide.EngineConfig[*World]{
		PopulationSize: 20,
		MaxGenerations: 5,
		FitnessFunc:    fitnessFunc,
		Selector:       nucleotide.GenericTournamentSelector[*World]{Size: 3},
		Crossoverer:    nucleotide.SinglePointCrossover{},
		Mutator:        nucleotide.CategoricalMutator{Probability: 0.1},
		Elitism:        1,
		Env:            &World{FoodAvailable: 10},
	}
	engine := nucleotide.NewEngine[*World](config)

	// 4. Run evolution
	best, _ := engine.Run(def)

	fmt.Printf("\nEvolution finished!\n")
	fmt.Printf("Best individual fitness: %.1f\n", best.Fitness)
	
	// 5. Save the winner for production
	filename := "best_genome.json"
	fmt.Printf("Saving winner to %s...\n", filename)
	if err := best.Save(filename); err != nil {
		fmt.Printf("Error saving: %v\n", err)
	}

	// 6. Simulate Production Deployment (CLEAN START)
	fmt.Println("\n--- Production Deployment Simulation ---")
	
	// Create a NEW definition variable for production
	// This simulates a fresh process start where the definitions are recreated.
	prodDef := nucleotide.NewDefinition[*World]()
	
	// The production code must define the same Loci and Genes with same IDs
	prodBehavior := prodDef.AddLocus("Behavior", nucleotide.LocusBehavioral)
	prodBehavior.AddGene("Glutton", func(ctx nucleotide.Context[*World]) {
		if ctx.Env.FoodAvailable > 0 {
			ctx.Env.FoodAvailable -= 2
		}
	})
	prodBehavior.AddGene("Frugal", func(ctx nucleotide.Context[*World]) {
		if ctx.Env.FoodAvailable > 0 {
			ctx.Env.FoodAvailable -= 1
		}
	})

	// Load the genome using the NEW definition
	fmt.Printf("Loading genome from %s using production definitions...\n", filename)
	loadedGenome, err := nucleotide.LoadGenome(prodDef, filename)
	if err != nil {
		fmt.Printf("Error loading: %v\n", err)
		return
	}
	
	prodIndividual := nucleotide.NewIndividual[*World](loadedGenome)
	prodWorld := &World{FoodAvailable: 100}
	
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()
	
	fmt.Println("Executing production individual...")
	prodIndividual.Express(ctx, prodWorld)
	fmt.Printf("Execution complete. Final Food: %d\n", prodWorld.FoodAvailable)
}
