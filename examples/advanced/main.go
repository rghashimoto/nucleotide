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

// AgentState holds individual agent variables during expression.
type AgentState struct {
	Energy int
}

func main() {
	// 1. Define the genome architecture for evolution
	def := nucleotide.NewDefinition[*World, AgentState]()

	// Gather Locus (Behavioral) - adds energy to AgentState by consuming food from environment
	gather := def.AddLocus("Gather", nucleotide.LocusBehavioral)
	gather.AddGene("GatherGlutton", func(ctx nucleotide.Context[*World, AgentState]) {
		if ctx.Env.FoodAvailable >= 2 {
			ctx.Env.FoodAvailable -= 2
			ctx.Individual.State.Energy += 4
		}
	})
	gather.AddGene("GatherFrugal", func(ctx nucleotide.Context[*World, AgentState]) {
		if ctx.Env.FoodAvailable >= 1 {
			ctx.Env.FoodAvailable -= 1
			ctx.Individual.State.Energy += 2
		}
	})

	// Metabolism Locus (Behavioral) - consumes energy stored in AgentState
	metabolism := def.AddLocus("Metabolism", nucleotide.LocusBehavioral)
	metabolism.AddGene("DigestFast", func(ctx nucleotide.Context[*World, AgentState]) {
		if ctx.Individual.State.Energy >= 3 {
			ctx.Individual.State.Energy -= 3
		}
	})
	metabolism.AddGene("DigestEfficient", func(ctx nucleotide.Context[*World, AgentState]) {
		if ctx.Individual.State.Energy >= 1 {
			ctx.Individual.State.Energy -= 1
		}
	})

	// 2. Define fitness function
	fitnessFunc := func(g nucleotide.Genome, env *World) float64 {
		cg := g.(*nucleotide.CategoricalGenome[*World, AgentState])
		localEnv := &World{FoodAvailable: env.FoodAvailable}
		ind := nucleotide.NewIndividual[*World, AgentState](cg)
		
		// Initialize the individual's state before running
		ind.State = AgentState{Energy: 0}
		
		ind.Express(context.Background(), localEnv)
		
		score := float64(ind.State.Energy) * 2.0
		if localEnv.FoodAvailable >= 0 {
			score += float64(localEnv.FoodAvailable)
		}
		// Prefer Frugal (index 1 is Gather, gene index 1 is GatherFrugal)
		if cg.GeneIndices[1] == 1 {
			score += 10.0
		}
		return score
	}

	// 3. Initialize engine
	config := nucleotide.EngineConfig[*World, AgentState]{
		PopulationSize: 20,
		MaxGenerations: 5,
		FitnessFunc:    fitnessFunc,
		Selector:       nucleotide.GenericTournamentSelector[*World, AgentState]{Size: 3},
		Crossoverers:   []nucleotide.Crossoverer{nucleotide.SinglePointCrossover{}},
		Mutators:       []nucleotide.Mutator{nucleotide.CategoricalMutator{Probability: 0.1}},
		Elitism:        1,
		Env:            &World{FoodAvailable: 10},
	}
	engine, err := nucleotide.NewEngine[*World, AgentState](config)
	if err != nil {
		panic(err)
	}

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
	prodDef := nucleotide.NewDefinition[*World, AgentState]()
	
	prodGather := prodDef.AddLocus("Gather", nucleotide.LocusBehavioral)
	prodGather.AddGene("GatherGlutton", func(ctx nucleotide.Context[*World, AgentState]) {
		if ctx.Env.FoodAvailable >= 2 {
			ctx.Env.FoodAvailable -= 2
			ctx.Individual.State.Energy += 4
		}
	})
	prodGather.AddGene("GatherFrugal", func(ctx nucleotide.Context[*World, AgentState]) {
		if ctx.Env.FoodAvailable >= 1 {
			ctx.Env.FoodAvailable -= 1
			ctx.Individual.State.Energy += 2
		}
	})

	prodMetabolism := prodDef.AddLocus("Metabolism", nucleotide.LocusBehavioral)
	prodMetabolism.AddGene("DigestFast", func(ctx nucleotide.Context[*World, AgentState]) {
		if ctx.Individual.State.Energy >= 3 {
			ctx.Individual.State.Energy -= 3
		}
	})
	prodMetabolism.AddGene("DigestEfficient", func(ctx nucleotide.Context[*World, AgentState]) {
		if ctx.Individual.State.Energy >= 1 {
			ctx.Individual.State.Energy -= 1
		}
	})

	// Load the genome using the NEW definition
	fmt.Printf("Loading genome from %s using production definitions...\n", filename)
	loadedGenome, err := nucleotide.LoadGenome(prodDef, filename)
	if err != nil {
		fmt.Printf("Error loading: %v\n", err)
		return
	}
	
	prodIndividual := nucleotide.NewIndividual[*World, AgentState](loadedGenome)
	prodIndividual.State = AgentState{Energy: 5} // Let's initialize with 5 energy points in production
	prodWorld := &World{FoodAvailable: 100}
	
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()
	
	fmt.Println("Executing production individual...")
	prodIndividual.Express(ctx, prodWorld)
	fmt.Printf("Execution complete. Final Food: %d, Final Agent Energy: %d\n", prodWorld.FoodAvailable, prodIndividual.State.Energy)
}
