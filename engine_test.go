package nucleotide

import (
	"context"
	"testing"
)

func TestEngine_Run_Categorical(t *testing.T) {
	def := NewDefinition[TestEnv, struct{}]()
	l1 := def.AddLocus("L1", LocusBehavioral)
	l1.AddGene("G1", func(ctx Context[TestEnv, struct{}]) {})
	
	config := EngineConfig[TestEnv, struct{}]{
		PopulationSize: 10,
		MaxGenerations: 2,
		FitnessFunc: func(g Genome, env TestEnv) []float64 {
			return []float64{1.0}
		},
		Selector:     GenericTournamentSelector[TestEnv, struct{}]{Size: 3},
		Crossoverers: []WeightedCrossoverer{{Crossoverer: SinglePointCrossover{}}},
		Mutators:     []WeightedMutator{{Mutator: CategoricalMutator{Probability: 0.1}}},
		Elitism:      1,
	}
	engine, err := NewEngine[TestEnv, struct{}](config)
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}
	best, err := engine.Run(def)
	if err != nil {
		t.Fatalf("Engine.Run failed: %v", err)
	}
	bestGenome := best.Genome.(*CategoricalGenome[TestEnv, struct{}])
	if bestGenome == nil {
		t.Fatal("Engine.Run returned nil best")
	}
	if engine.Generation != 2 {
		t.Errorf("Expected 2 generations, got %d", engine.Generation)
	}
}

func TestExpress_Cancellation(t *testing.T) {
	def := NewDefinition[TestEnv, struct{}]()
	l1 := def.AddLocus("L1", LocusBehavioral)
	count := 0
	l1.AddGene("G1", func(ctx Context[TestEnv, struct{}]) { count++ })
	l1.AddGene("G2", func(ctx Context[TestEnv, struct{}]) { count++ })
	
	g := &CategoricalGenome[TestEnv, struct{}]{
		Definition:  def,
		GeneIndices: []int{0, 0, 1}, // Seq, G1, G2
	}
	ind := NewIndividual[TestEnv, struct{}](g)
	
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately
	
	ind.Express(ctx, TestEnv{})
	if count > 0 {
		t.Errorf("Express did not respect cancellation: executed %d genes", count)
	}
}

func TestExpress_StateInteraction(t *testing.T) {
	def := NewDefinition[TestEnv, *CounterState]()
	
	l1 := def.AddLocus("L1", LocusBehavioral)
	// First gene increments state Count
	l1.AddGene("G1", func(ctx Context[TestEnv, *CounterState]) {
		ctx.Individual.State.Count += 5
	})
	
	l2 := def.AddLocus("L2", LocusBehavioral)
	// Second gene multiplies state Count by 2
	l2.AddGene("G2", func(ctx Context[TestEnv, *CounterState]) {
		ctx.Individual.State.Count *= 2
	})
	
	g := &CategoricalGenome[TestEnv, *CounterState]{
		Definition:  def,
		GeneIndices: []int{0, 0, 0}, // Execution Order (Sequential), L1 (G1), L2 (G2)
	}
	
	ind := NewIndividual[TestEnv, *CounterState](g)
	state := &CounterState{Count: 3}
	ind.State = state
	
	ind.Express(context.Background(), TestEnv{})
	
	// Value should be: (3 + 5) * 2 = 16
	if state.Count != 16 {
		t.Errorf("Expected Count to be 16, got %d. Gene execution state interaction failed.", state.Count)
	}
}

func TestNewEngine_MissingFitnessFunc(t *testing.T) {
	config := EngineConfig[TestEnv, struct{}]{
		PopulationSize: 10,
		MaxGenerations: 2,
	}
	_, err := NewEngine[TestEnv, struct{}](config)
	if err == nil {
		t.Error("Expected error when FitnessFunc is not defined, got nil")
	} else if err.Error() != "FitnessFunc must be defined in EngineConfig" {
		t.Errorf("Expected 'FitnessFunc must be defined in EngineConfig', got '%v'", err)
	}
}

func TestNewEngine_DefaultFallbacks(t *testing.T) {
	config := EngineConfig[TestEnv, struct{}]{
		PopulationSize: 10,
		MaxGenerations: 2,
		FitnessFunc: func(g Genome, env TestEnv) []float64 {
			return []float64{1.0}
		},
	}
	engine, err := NewEngine[TestEnv, struct{}](config)
	if err != nil {
		t.Fatalf("Failed to create engine with default fallbacks: %v", err)
	}

	if engine.Config.Selector == nil {
		t.Error("Expected default Selector to be set, got nil")
	} else {
		if _, ok := engine.Config.Selector.(GenericTournamentSelector[TestEnv, struct{}]); !ok {
			t.Errorf("Expected default Selector to be GenericTournamentSelector, got %T", engine.Config.Selector)
		}
	}

	if len(engine.Config.Crossoverers) != 1 {
		t.Errorf("Expected 1 default crossoverer, got %d", len(engine.Config.Crossoverers))
	} else {
		if _, ok := engine.Config.Crossoverers[0].Crossoverer.(DefaultCrossoverer); !ok {
			t.Errorf("Expected default crossoverer to be DefaultCrossoverer, got %T", engine.Config.Crossoverers[0].Crossoverer)
		}
	}

	if len(engine.Config.Mutators) != 1 {
		t.Errorf("Expected 1 default mutator, got %d", len(engine.Config.Mutators))
	} else {
		if _, ok := engine.Config.Mutators[0].Mutator.(DefaultMutator); !ok {
			t.Errorf("Expected default mutator to be DefaultMutator, got %T", engine.Config.Mutators[0].Mutator)
		}
	}
}

func TestEngine_RoundRobinOperators(t *testing.T) {
	config := EngineConfig[TestEnv, struct{}]{
		PopulationSize: 10,
		MaxGenerations: 2,
		FitnessFunc: func(g Genome, env TestEnv) []float64 {
			return []float64{1.0}
		},
		Crossoverers: []WeightedCrossoverer{
			{Crossoverer: MockCrossoverer{id: 1}},
			{Crossoverer: MockCrossoverer{id: 2}},
		},
		Mutators: []WeightedMutator{
			{Mutator: MockMutator{id: 1}},
			{Mutator: MockMutator{id: 2}},
			{Mutator: MockMutator{id: 3}},
		},
	}
	engine, err := NewEngine[TestEnv, struct{}](config)
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}

	// Verify crossover round-robin sequencing
	c1 := engine.selectCrossoverer().(MockCrossoverer)
	c2 := engine.selectCrossoverer().(MockCrossoverer)
	c3 := engine.selectCrossoverer().(MockCrossoverer)
	c4 := engine.selectCrossoverer().(MockCrossoverer)

	if c1.id != 1 || c2.id != 2 || c3.id != 1 || c4.id != 2 {
		t.Errorf("Expected crossoverer sequencing [1, 2, 1, 2], got [%d, %d, %d, %d]", c1.id, c2.id, c3.id, c4.id)
	}

	// Verify mutator round-robin sequencing
	m1 := engine.selectMutator().(MockMutator)
	m2 := engine.selectMutator().(MockMutator)
	m3 := engine.selectMutator().(MockMutator)
	m4 := engine.selectMutator().(MockMutator)

	if m1.id != 1 || m2.id != 2 || m3.id != 3 || m4.id != 1 {
		t.Errorf("Expected mutator sequencing [1, 2, 3, 1], got [%d, %d, %d, %d]", m1.id, m2.id, m3.id, m4.id)
	}
}

func TestEngine_WeightedOperators(t *testing.T) {
	config := EngineConfig[TestEnv, struct{}]{
		PopulationSize: 10,
		MaxGenerations: 2,
		FitnessFunc: func(g Genome, env TestEnv) []float64 {
			return []float64{1.0}
		},
		Crossoverers: []WeightedCrossoverer{
			{Crossoverer: MockCrossoverer{id: 1}, Weight: 0.0},
			{Crossoverer: MockCrossoverer{id: 2}, Weight: 1.0},
		},
		Mutators: []WeightedMutator{
			{Mutator: MockMutator{id: 1}, Weight: 1.0},
			{Mutator: MockMutator{id: 2}, Weight: 0.0},
		},
	}
	engine, err := NewEngine[TestEnv, struct{}](config)
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}

	// With weights [0.0, 1.0], only crossoverer 2 should be selected.
	for i := 0; i < 100; i++ {
		c := engine.selectCrossoverer().(MockCrossoverer)
		if c.id != 2 {
			t.Errorf("Expected crossoverer with weight 1.0 to always be selected, but got id %d", c.id)
			break
		}
	}

	// With weights [1.0, 0.0], only mutator 1 should be selected.
	for i := 0; i < 100; i++ {
		m := engine.selectMutator().(MockMutator)
		if m.id != 1 {
			t.Errorf("Expected mutator with weight 1.0 to always be selected, but got id %d", m.id)
			break
		}
	}

	// Test invalid negative weights validation inside NewEngine
	invalidConfig := config
	invalidConfig.Crossoverers = []WeightedCrossoverer{
		{Crossoverer: MockCrossoverer{id: 1}, Weight: -1.0},
		{Crossoverer: MockCrossoverer{id: 2}, Weight: 2.0},
	}
	_, err = NewEngine[TestEnv, struct{}](invalidConfig)
	if err == nil {
		t.Error("Expected error for negative Crossoverer weight, got nil")
	}
}

func TestElitism_TopN(t *testing.T) {
	pop := Population[TestEnv, struct{}]{
		{Fitness: []float64{10}, Genome: BitGenome{false}},
		{Fitness: []float64{30}, Genome: BitGenome{true}},
		{Fitness: []float64{20}, Genome: BitGenome{false}},
	}
	elites := TopNElitism(pop, 2)
	if len(elites) != 2 {
		t.Fatalf("Expected 2 elites, got %d", len(elites))
	}
	if elites[0].Genome.(BitGenome)[0] != true {
		t.Errorf("TopNElitism failed to select the best individual (Fitness 30)")
	}
	if elites[1].Genome.(BitGenome)[0] != false {
		t.Errorf("TopNElitism failed to select the second best individual (Fitness 20)")
	}
}

func TestEngine_DefaultPopulationSizeDeduction(t *testing.T) {
	// L1 has 2 genes, L2 has 3 genes.
	// Formula: 40 * 2 * 3 = 240
	def := NewDefinition[TestEnv, struct{}]()
	l1 := def.AddLocus("L1", LocusBehavioral)
	l1.AddGene("G1", func(ctx Context[TestEnv, struct{}]) {})
	l1.AddGene("G2", func(ctx Context[TestEnv, struct{}]) {})

	l2 := def.AddLocus("L2", LocusBehavioral)
	l2.AddGene("G1", func(ctx Context[TestEnv, struct{}]) {})
	l2.AddGene("G2", func(ctx Context[TestEnv, struct{}]) {})
	l2.AddGene("G3", func(ctx Context[TestEnv, struct{}]) {})

	config := EngineConfig[TestEnv, struct{}]{
		PopulationSize: 0, // Request dynamic deduction
		MaxGenerations: 1,
		FitnessFunc: func(g Genome, env TestEnv) []float64 {
			return []float64{1.0}
		},
		Selector: GenericTournamentSelector[TestEnv, struct{}]{Size: 3},
	}

	engine, err := NewEngine[TestEnv, struct{}](config)
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}

	if engine.Config.PopulationSize != 0 {
		t.Errorf("Expected PopulationSize to remain 0 before running, got %d", engine.Config.PopulationSize)
	}

	_, err = engine.Run(def)
	if err != nil {
		t.Fatalf("Engine.Run failed: %v", err)
	}

	// Dynamic deduction should have computed 40 * 2 * 3 = 240
	if engine.Config.PopulationSize != 240 {
		t.Errorf("Expected dynamically deduced PopulationSize to be 240, got %d", engine.Config.PopulationSize)
	}

	if len(engine.Population) != 240 {
		t.Errorf("Expected Population size to be initialized to 240, got %d", len(engine.Population))
	}
}

