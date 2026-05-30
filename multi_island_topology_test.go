package nucleotide

import (
	"testing"
)

type TopologyTestEnv struct {
	Value float64
}

type TopologyTestState struct{}

func TestTorusFactorization(t *testing.T) {
	// Test factorization results for various counts
	tests := []struct {
		n int
		w int
		h int
	}{
		{9, 3, 3},
		{16, 4, 4},
		{8, 2, 4},
		{6, 2, 3},
		{7, 1, 7}, // prime fallback
		{13, 1, 13}, // prime fallback
	}

	for _, tt := range tests {
		w, h := getTorusDimensions(tt.n)
		if w != tt.w || h != tt.h {
			t.Errorf("For n=%d, expected grid %dx%d, got %dx%d", tt.n, tt.w, tt.h, w, h)
		}
	}
}

func TestTopology_Torus(t *testing.T) {
	// Create a 4-island Engine using Torus topology (factors into 2x2 grid)
	def := NewDefinition[TopologyTestEnv, TopologyTestState]()
	l := def.AddLocus("L1", LocusBehavioral)
	l.AddGene("G1", func(ctx Context[TopologyTestEnv, TopologyTestState]) {})

	config := EngineConfig[TopologyTestEnv, TopologyTestState]{
		PopulationSize: 4,
		FitnessFunc: func(g Genome, env TopologyTestEnv) []float64 {
			return []float64{env.Value}
		},
		Selector: GenericTournamentSelector[TopologyTestEnv, TopologyTestState]{Size: 2},
	}

	miConfig := MultiIslandEngineConfig[TopologyTestEnv, TopologyTestState]{
		NumIslands:        4,
		MigrationInterval: 5,
		MigrationRate:     1,
		MigrationTopology: TopologyTorus,
		MigrationPolicy:   PolicyBestReplaceWorst,
		EngineConfig:      config,
		EnvFactory: func(islandIdx int) TopologyTestEnv {
			return TopologyTestEnv{Value: float64((islandIdx + 1) * 10)}
		},
	}

	miEngine, err := NewMultiIslandEngine(miConfig)
	if err != nil {
		t.Fatalf("Failed to create MultiIslandEngine with Torus: %v", err)
	}

	// Initialize populations manually with recognizable distinct fitness values
	for i, island := range miEngine.Islands {
		island.Population = make(Population[TopologyTestEnv, TopologyTestState], 4)
		for p := 0; p < 4; p++ {
			// All individuals on island i have fitness (i+1)*10
			val := float64((i + 1) * 10)
			island.Population[p] = &Individual[TopologyTestEnv, TopologyTestState]{
				Fitness: []float64{val},
				Genome:  BitGenome{true},
			}
		}
	}

	// Verify neighbors for Island 0 in 2x2 Torus grid:
	// x=0, y=0.
	// North: y-1 wraps to 1 -> idx = 1*2+0 = 2
	// South: y+1 wraps to 1 -> idx = 1*2+0 = 2
	// East: x+1 wraps to 1 -> idx = 0*2+1 = 1
	// West: x-1 wraps to 1 -> idx = 0*2+1 = 1
	// Deduplicated: neighbors should be 1 and 2 (since 2x2 wraps duplicate directions)
	nbrs := miEngine.getNeighbors(0)
	if len(nbrs) != 2 {
		t.Errorf("Expected deduplicated Torus neighbors for island 0 on 2x2 grid to be 2, got %d", len(nbrs))
	}

	// Trigger migration
	miEngine.migrate()

	// Verify that Island 0 received migrants from neighbors 1 and 2
	// (Its population of 10.0s should have worst elements replaced by 20.0s and 30.0s)
	has20 := false
	has30 := false
	for _, ind := range miEngine.Islands[0].Population {
		if len(ind.Fitness) > 0 {
			if ind.Fitness[0] == 20.0 {
				has20 = true
			}
			if ind.Fitness[0] == 30.0 {
				has30 = true
			}
		}
	}

	if !has20 || !has30 {
		t.Errorf("Torus migration failed to route correct cloned individuals: has20=%t, has30=%t", has20, has30)
	}
}

func TestTopology_Hypercube(t *testing.T) {
	// 1. Verify invalid island counts trigger early validation errors
	config := EngineConfig[TopologyTestEnv, TopologyTestState]{
		PopulationSize: 4,
		FitnessFunc: func(g Genome, env TopologyTestEnv) []float64 {
			return []float64{env.Value}
		},
		Selector: GenericTournamentSelector[TopologyTestEnv, TopologyTestState]{Size: 2},
	}

	invalidConfig := MultiIslandEngineConfig[TopologyTestEnv, TopologyTestState]{
		NumIslands:        6, // not a power of 2
		MigrationTopology: TopologyHypercube,
		EngineConfig:      config,
	}

	_, err := NewMultiIslandEngine(invalidConfig)
	if err == nil {
		t.Error("Expected error during NewMultiIslandEngine initialization with non-power of 2 hypercube, got nil")
	}

	// 2. Setup a valid 8-island hypercube (dim = 3)
	validConfig := MultiIslandEngineConfig[TopologyTestEnv, TopologyTestState]{
		NumIslands:        8,
		MigrationInterval: 5,
		MigrationRate:     1,
		MigrationTopology: TopologyHypercube,
		MigrationPolicy:   PolicyBestReplaceWorst,
		EngineConfig:      config,
		EnvFactory: func(islandIdx int) TopologyTestEnv {
			return TopologyTestEnv{Value: float64(islandIdx + 1)}
		},
	}

	miEngine, err := NewMultiIslandEngine(validConfig)
	if err != nil {
		t.Fatalf("Failed to create MultiIslandEngine with valid hypercube: %v", err)
	}

	// Verify neighbors of Island 0 (bin: 000). Neighbors flip 1 bit:
	// - 000 ^ 001 = 001 (1)
	// - 000 ^ 010 = 010 (2)
	// - 000 ^ 100 = 100 (4)
	nbrs := miEngine.getNeighbors(0)
	if len(nbrs) != 3 {
		t.Fatalf("Expected 3 neighbors for 3D hypercube island 0, got %d", len(nbrs))
	}

	nbrMap := make(map[int]bool)
	for _, n := range nbrs {
		nbrMap[n] = true
	}

	if !nbrMap[1] || !nbrMap[2] || !nbrMap[4] {
		t.Errorf("Hypercube neighbors mapped incorrectly: expected [1, 2, 4], got %v", nbrs)
	}
}

func TestTopology_Star(t *testing.T) {
	config := EngineConfig[TopologyTestEnv, TopologyTestState]{
		PopulationSize: 4,
		FitnessFunc: func(g Genome, env TopologyTestEnv) []float64 {
			return []float64{env.Value}
		},
		Selector: GenericTournamentSelector[TopologyTestEnv, TopologyTestState]{Size: 2},
	}

	miConfig := MultiIslandEngineConfig[TopologyTestEnv, TopologyTestState]{
		NumIslands:        4,
		MigrationInterval: 5,
		MigrationRate:     1,
		MigrationTopology: TopologyStar,
		MigrationPolicy:   PolicyBestReplaceWorst,
		EngineConfig:      config,
		EnvFactory: func(islandIdx int) TopologyTestEnv {
			return TopologyTestEnv{Value: float64((islandIdx + 1) * 10)}
		},
	}

	miEngine, err := NewMultiIslandEngine(miConfig)
	if err != nil {
		t.Fatalf("Failed to create Star engine: %v", err)
	}

	// Initialize populations manually
	for i, island := range miEngine.Islands {
		island.Population = make(Population[TopologyTestEnv, TopologyTestState], 4)
		for p := 0; p < 4; p++ {
			island.Population[p] = &Individual[TopologyTestEnv, TopologyTestState]{
				Fitness: []float64{float64((i + 1) * 10)},
				Genome:  BitGenome{true},
			}
		}
	}

	// Verify neighbors
	// Hub (0) -> [1, 2, 3]
	hubNbrs := miEngine.getNeighbors(0)
	if len(hubNbrs) != 3 {
		t.Errorf("Expected hub to connect to 3 spokes, got %d", len(hubNbrs))
	}

	// Spoke (1) -> [0]
	spokeNbrs := miEngine.getNeighbors(1)
	if len(spokeNbrs) != 1 || spokeNbrs[0] != 0 {
		t.Errorf("Expected spoke to connect only to hub (0), got %v", spokeNbrs)
	}

	// Run migration
	miEngine.migrate()

	// Verify Hub (Island 0) has received migrants from all spokes (10.0 replaced by 20.0, 30.0, 40.0)
	has20 := false
	has30 := false
	has40 := false
	for _, ind := range miEngine.Islands[0].Population {
		if len(ind.Fitness) > 0 {
			switch ind.Fitness[0] {
			case 20.0:
				has20 = true
			case 30.0:
				has30 = true
			case 40.0:
				has40 = true
			}
		}
	}

	if !has20 || !has30 || !has40 {
		t.Errorf("Star hub did not receive all spoke migrants: has20=%t, has30=%t, has40=%t", has20, has30, has40)
	}

	// Verify Spokes (Islands 1, 2, 3) have received migrants from Hub (Island 0) which had fitness 10.0
	for spokeIdx := 1; spokeIdx <= 3; spokeIdx++ {
		has10 := false
		for _, ind := range miEngine.Islands[spokeIdx].Population {
			if len(ind.Fitness) > 0 && ind.Fitness[0] == 10.0 {
				has10 = true
				break
			}
		}
		if !has10 {
			t.Errorf("Spoke island %d did not receive elite hub migrant", spokeIdx)
		}
	}
}

func TestTopology_BufferIntegration(t *testing.T) {
	// Verify that if many incoming migrants arrive, they are correctly filtered/integrated without exceeding population sizes
	config := EngineConfig[TopologyTestEnv, TopologyTestState]{
		PopulationSize: 2, // small pop
		FitnessFunc: func(g Genome, env TopologyTestEnv) []float64 {
			return []float64{env.Value}
		},
		Selector: GenericTournamentSelector[TopologyTestEnv, TopologyTestState]{Size: 2},
	}

	// 5 islands, with Star topology: Hub gets 4 incoming migrants in a single cycle (4 spokes)
	// Since Hub pop size is 2, it should successfully integrate the best 2 spoke migrants, and not panic/leak
	miConfig := MultiIslandEngineConfig[TopologyTestEnv, TopologyTestState]{
		NumIslands:        5,
		MigrationInterval: 5,
		MigrationRate:     1,
		MigrationTopology: TopologyStar,
		MigrationPolicy:   PolicyBestReplaceWorst,
		EngineConfig:      config,
	}

	miEngine, err := NewMultiIslandEngine(miConfig)
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}

	// Set Hub pop fitness to [1.0, 2.0]
	miEngine.Islands[0].Population = Population[TopologyTestEnv, TopologyTestState]{
		{Fitness: []float64{1.0}, Genome: BitGenome{true}},
		{Fitness: []float64{2.0}, Genome: BitGenome{true}},
	}

	// Set Spoke pops so each has 1 elite individual with higher fitness
	// Spoke 1: 10.0, Spoke 2: 20.0, Spoke 3: 30.0, Spoke 4: 40.0
	for idx := 1; idx <= 4; idx++ {
		miEngine.Islands[idx].Population = Population[TopologyTestEnv, TopologyTestState]{
			{Fitness: []float64{float64(idx * 10)}, Genome: BitGenome{true}},
			{Fitness: []float64{float64(idx * 10)}, Genome: BitGenome{true}},
		}
	}

	// Migrate
	miEngine.migrate()

	// Hub population should now have size 2 (not leaked)
	hubPop := miEngine.Islands[0].Population
	if len(hubPop) != 2 {
		t.Fatalf("Expected Hub population size 2 after buffer integration, got %d", len(hubPop))
	}

	// Hub population should have absorbed the best two incoming migrants (30.0 and 40.0)
	// replacing its original worst elements (1.0 and 2.0)
	has30 := false
	has40 := false
	for _, ind := range hubPop {
		if len(ind.Fitness) > 0 {
			if ind.Fitness[0] == 30.0 {
				has30 = true
			}
			if ind.Fitness[0] == 40.0 {
				has40 = true
			}
		}
	}

	if !has30 || !has40 {
		t.Errorf("Buffer integration failed to merge/select the best incoming migrants: pop=%+v", hubPop)
	}
}
