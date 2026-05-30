package nucleotide

import (
	"testing"
)

type MultiIslandTestEnv struct {
	IslandID int
	Value    float64
}

func TestMultiIslandEngine_BasicRingTopology(t *testing.T) {
	// Setup simple categorical optimization
	def := NewDefinition[MultiIslandTestEnv, struct{}]()
	l1 := def.AddLocus("L1", LocusBehavioral)
	l1.AddGene("G1", func(ctx Context[MultiIslandTestEnv, struct{}]) {})
	l1.AddGene("G2", func(ctx Context[MultiIslandTestEnv, struct{}]) {})

	fitnessFunc := func(g Genome, env MultiIslandTestEnv) []float64 {
		cg := g.(*CategoricalGenome[MultiIslandTestEnv, struct{}])
		score := 0.0
		if cg.GeneIndices[1] == 1 {
			score += 10.0 // G2 is ideal
		}
		return []float64{score}
	}

	config := EngineConfig[MultiIslandTestEnv, struct{}]{
		PopulationSize: 10,
		MaxGenerations: 15,
		FitnessFunc:    fitnessFunc,
		Selector:       GenericTournamentSelector[MultiIslandTestEnv, struct{}]{Size: 2},
		Elitism:        1,
		Env:            MultiIslandTestEnv{IslandID: 99},
	}

	miConfig := MultiIslandEngineConfig[MultiIslandTestEnv, struct{}]{
		NumIslands:        3,
		MigrationInterval: 5,
		MigrationRate:     1,
		MigrationTopology: TopologyRing,
		MigrationPolicy:   PolicyBestReplaceWorst,
		EngineConfig:      config,
	}

	miEngine, err := NewMultiIslandEngine(miConfig)
	if err != nil {
		t.Fatalf("Failed to create MultiIslandEngine: %v", err)
	}

	if len(miEngine.Islands) != 3 {
		t.Errorf("Expected 3 islands, got %d", len(miEngine.Islands))
	}

	// Verify Option A (Shared Env)
	for i, island := range miEngine.Islands {
		if island.Config.Env.IslandID != 99 {
			t.Errorf("Island %d environment not shared correctly: expected 99, got %d", i, island.Config.Env.IslandID)
		}
	}

	best, err := miEngine.Run(def)
	if err != nil {
		t.Fatalf("MultiIslandEngine Run failed: %v", err)
	}

	if best == nil {
		t.Fatal("Run returned nil best individual")
	}

	// Verify that the global best individual has high fitness
	if len(best.Fitness) == 0 || best.Fitness[0] < 10.0 {
		t.Errorf("Expected fitness 10.0, got %v", best.Fitness)
	}
}

func TestMultiIslandEngine_DistributedEnvs(t *testing.T) {
	def := NewDefinition[MultiIslandTestEnv, struct{}]()
	l1 := def.AddLocus("L1", LocusBehavioral)
	l1.AddGene("G1", func(ctx Context[MultiIslandTestEnv, struct{}]) {})

	fitnessFunc := func(g Genome, env MultiIslandTestEnv) []float64 {
		// Return IslandID as fitness to verify environment specialization
		return []float64{float64(env.IslandID)}
	}

	config := EngineConfig[MultiIslandTestEnv, struct{}]{
		PopulationSize: 6,
		MaxGenerations: 6,
		FitnessFunc:    fitnessFunc,
		Selector:       GenericTournamentSelector[MultiIslandTestEnv, struct{}]{Size: 2},
		Elitism:        1,
	}

	// Factory for Option B & C: Distributed/Heterogeneous Envs
	envFactory := func(islandIdx int) MultiIslandTestEnv {
		return MultiIslandTestEnv{
			IslandID: islandIdx + 1, // Island 1 gets ID 1, Island 2 gets ID 2
		}
	}

	miConfig := MultiIslandEngineConfig[MultiIslandTestEnv, struct{}]{
		NumIslands:        2,
		MigrationInterval: 2,
		MigrationRate:     1,
		MigrationTopology: TopologyRandom,
		MigrationPolicy:   PolicyRandomReplaceRandom,
		EngineConfig:      config,
		EnvFactory:        envFactory,
	}

	miEngine, err := NewMultiIslandEngine(miConfig)
	if err != nil {
		t.Fatalf("Failed to create MultiIslandEngine: %v", err)
	}

	// Verify distributed env allocation
	if miEngine.Islands[0].Config.Env.IslandID != 1 {
		t.Errorf("Island 0 expected ID 1, got %d", miEngine.Islands[0].Config.Env.IslandID)
	}
	if miEngine.Islands[1].Config.Env.IslandID != 2 {
		t.Errorf("Island 1 expected ID 2, got %d", miEngine.Islands[1].Config.Env.IslandID)
	}

	best, err := miEngine.Run(def)
	if err != nil {
		t.Fatalf("MultiIslandEngine Run failed: %v", err)
	}

	if best == nil {
		t.Fatal("Run returned nil best individual")
	}

	// The global best individual should have fitness of the highest environment value (which is 2.0 on Island 2)
	if len(best.Fitness) == 0 || best.Fitness[0] != 2.0 {
		t.Errorf("Expected global best fitness to be 2.0, got %v", best.Fitness)
	}
}

func TestMultiIslandEngine_InvalidConfig(t *testing.T) {
	_, err := NewMultiIslandEngine(MultiIslandEngineConfig[MultiIslandTestEnv, struct{}]{
		NumIslands: 0,
	})
	if err == nil {
		t.Error("Expected error when NumIslands is <= 0, got nil")
	}
}
