package nucleotide

import (
	"context"
	"math"
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

type CustomMutatorForTest struct {
	Probability float64
	OtherField  string
}

func (m CustomMutatorForTest) Mutate(g Genome) Genome {
	return g
}

func TestEngine_ScaleMutator(t *testing.T) {
	config := EngineConfig[TestEnv, struct{}]{
		FitnessFunc: func(g Genome, env TestEnv) []float64 {
			return []float64{1.0}
		},
	}
	engine, err := NewEngine[TestEnv, struct{}](config)
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}

	// 1. Test built-in CategoricalMutator
	catMut := CategoricalMutator{Probability: 0.1}
	scaledCat := engine.scaleMutator(catMut, 2.5).(CategoricalMutator)
	if scaledCat.Probability != 0.25 {
		t.Errorf("Expected scaled CategoricalMutator probability 0.25, got %f", scaledCat.Probability)
	}
	if catMut.Probability != 0.1 {
		t.Errorf("Original CategoricalMutator was mutated: %f", catMut.Probability)
	}

	// 2. Test pointer built-in CategoricalMutator
	catPtr := &CategoricalMutator{Probability: 0.1}
	scaledPtr := engine.scaleMutator(catPtr, 2.5).(*CategoricalMutator)
	if scaledPtr.Probability != 0.25 {
		t.Errorf("Expected scaled pointer CategoricalMutator probability 0.25, got %f", scaledPtr.Probability)
	}
	if catPtr.Probability != 0.1 {
		t.Errorf("Original pointer CategoricalMutator was mutated: %f", catPtr.Probability)
	}

	// 3. Test custom mutator reflection value
	customMut := CustomMutatorForTest{Probability: 0.2, OtherField: "test"}
	scaledCustom := engine.scaleMutator(customMut, 3.0).(CustomMutatorForTest)
	if math.Abs(scaledCustom.Probability-0.6) > 1e-9 {
		t.Errorf("Expected scaled CustomMutator probability 0.6, got %f", scaledCustom.Probability)
	}
	if scaledCustom.OtherField != "test" {
		t.Errorf("Expected CustomMutator OtherField to remain 'test', got '%s'", scaledCustom.OtherField)
	}

	// 4. Test custom mutator reflection pointer
	customPtr := &CustomMutatorForTest{Probability: 0.2, OtherField: "test"}
	scaledCustomPtr := engine.scaleMutator(customPtr, 3.0).(*CustomMutatorForTest)
	if math.Abs(scaledCustomPtr.Probability-0.6) > 1e-9 {
		t.Errorf("Expected scaled pointer CustomMutator probability 0.6, got %f", scaledCustomPtr.Probability)
	}
}

func TestEngine_GenotypicDiversity(t *testing.T) {
	config := EngineConfig[TestEnv, struct{}]{
		FitnessFunc: func(g Genome, env TestEnv) []float64 {
			return []float64{1.0}
		},
	}
	engine, err := NewEngine[TestEnv, struct{}](config)
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}

	// 1. BitGenome Diversity
	engine.Population = Population[TestEnv, struct{}]{
		NewIndividual[TestEnv, struct{}](BitGenome{true, false, true}),
		NewIndividual[TestEnv, struct{}](BitGenome{true, true, false}),
	}
	// Locus 0: true (2), dominance = 1.0, diversity = 0.0
	// Locus 1: false (1), true (1), dominance = 0.5, diversity = 0.5
	// Locus 2: true (1), false (1), dominance = 0.5, diversity = 0.5
	// Avg: (0.0 + 0.5 + 0.5) / 3 = 0.3333333333333333
	divBit := engine.genotypicDiversity()
	if math.Abs(divBit-0.333333333) > 1e-6 {
		t.Errorf("Expected BitGenome diversity ~0.333, got %f", divBit)
	}

	// 2. FloatGenome Diversity
	engine.Population = Population[TestEnv, struct{}]{
		NewIndividual[TestEnv, struct{}](FloatGenome{1.0, 2.0}),
		NewIndividual[TestEnv, struct{}](FloatGenome{1.0, 4.0}),
	}
	// Locus 0: min = 1, max = 1, max == min -> 0.0
	// Locus 1: min = 2, max = 4, mean = 3, stdDev = 1.0. locusDiv = 2 * 1 / 2 = 1.0
	// Avg: (0.0 + 1.0) / 2 = 0.5
	divFloat := engine.genotypicDiversity()
	if math.Abs(divFloat-0.5) > 1e-6 {
		t.Errorf("Expected FloatGenome diversity 0.5, got %f", divFloat)
	}
}

func TestEngine_AdaptiveMutation(t *testing.T) {
	def := NewDefinition[TestEnv, struct{}]()
	l1 := def.AddLocus("L1", LocusBehavioral)
	l1.AddGene("G1", func(ctx Context[TestEnv, struct{}]) {})
	l1.AddGene("G2", func(ctx Context[TestEnv, struct{}]) {})

	callbackTriggered := false
	var lastDiversity float64
	var lastScaler float64

	config := EngineConfig[TestEnv, struct{}]{
		PopulationSize:    10,
		MaxGenerations:    3,
		FitnessFunc: func(g Genome, env TestEnv) []float64 {
			return []float64{1.0}
		},
		AdaptiveMutation:  true,
		MaxMutationScaler: 4.0,
		OnMutationAdapted: func(generation int, diversity float64, currentScaler float64) {
			callbackTriggered = true
			lastDiversity = diversity
			lastScaler = currentScaler
		},
		Mutators: []WeightedMutator{{Mutator: CategoricalMutator{Probability: 0.1}}},
	}

	engine, err := NewEngine[TestEnv, struct{}](config)
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}

	_, err = engine.Run(def)
	if err != nil {
		t.Fatalf("Engine.Run failed: %v", err)
	}

	if len(engine.DiversityHistory) != 3 {
		t.Errorf("Expected 3 entries in DiversityHistory, got %d", len(engine.DiversityHistory))
	}

	if !callbackTriggered {
		t.Error("Expected OnMutationAdapted callback to be triggered, but it wasn't")
	}

	expectedScaler := 1.0 + (1.0-lastDiversity)*(4.0-1.0)
	if math.Abs(lastScaler-expectedScaler) > 1e-9 {
		t.Errorf("Expected lastScaler to be %f, got %f", expectedScaler, lastScaler)
	}
}

func TestEngine_AgeBiasedMutation(t *testing.T) {
	def := NewDefinition[TestEnv, struct{}]()
	l1 := def.AddLocus("L1", LocusBehavioral)
	l1.AddGene("G1", func(ctx Context[TestEnv, struct{}]) {})
	l1.AddGene("G2", func(ctx Context[TestEnv, struct{}]) {})

	config := EngineConfig[TestEnv, struct{}]{
		PopulationSize:       10,
		MaxGenerations:       2,
		FitnessFunc: func(g Genome, env TestEnv) []float64 {
			return []float64{1.0}
		},
		AgeBiasedMutation:    true,
		AgeMutationThreshold: 1,
		AgeMutationScaler:    5.0,
		Mutators:             []WeightedMutator{{Mutator: CategoricalMutator{Probability: 0.1}}},
	}

	engine, err := NewEngine[TestEnv, struct{}](config)
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}

	_, err = engine.Run(def)
	if err != nil {
		t.Fatalf("Engine.Run failed: %v", err)
	}

	if !engine.Config.AgeBiasedMutation {
		t.Error("Expected AgeBiasedMutation to be enabled in config")
	}
}

func TestEngine_ParallelFitness(t *testing.T) {
	def := NewDefinition[TestEnv, struct{}]()
	l1 := def.AddLocus("L1", LocusBehavioral)
	l1.AddGene("G1", func(ctx Context[TestEnv, struct{}]) {})
	l1.AddGene("G2", func(ctx Context[TestEnv, struct{}]) {})

	config := EngineConfig[TestEnv, struct{}]{
		PopulationSize:   20,
		MaxGenerations:   2,
		FitnessFunc: func(g Genome, env TestEnv) []float64 {
			return []float64{10.0}
		},
		ConcurrencyLimit: 4,
	}

	engine, err := NewEngine[TestEnv, struct{}](config)
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}

	best, err := engine.Run(def)
	if err != nil {
		t.Fatalf("Engine.Run failed under parallel execution: %v", err)
	}

	if best == nil {
		t.Fatal("Engine.Run returned nil best individual")
	}

	for _, ind := range engine.Population {
		if len(ind.Fitness) == 0 || ind.Fitness[0] != 10.0 {
			t.Errorf("Expected fitness 10.0, got %v", ind.Fitness)
		}
	}
}

func TestEngine_ParallelReproduction(t *testing.T) {
	def := NewDefinition[TestEnv, struct{}]()
	l1 := def.AddLocus("L1", LocusBehavioral)
	l1.AddGene("G1", func(ctx Context[TestEnv, struct{}]) {})
	l1.AddGene("G2", func(ctx Context[TestEnv, struct{}]) {})

	config := EngineConfig[TestEnv, struct{}]{
		PopulationSize:   30,
		MaxGenerations:   5,
		FitnessFunc: func(g Genome, env TestEnv) []float64 {
			return []float64{1.0}
		},
		ConcurrencyLimit: 4,
	}

	engine, err := NewEngine[TestEnv, struct{}](config)
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}

	best, err := engine.Run(def)
	if err != nil {
		t.Fatalf("Engine.Run failed under parallel reproduction: %v", err)
	}

	if len(engine.Population) != 30 {
		t.Errorf("Expected population size 30, got %d", len(engine.Population))
	}
	if best == nil {
		t.Fatal("Engine.Run returned nil best individual")
	}
}

func TestEngine_DisableParallelism(t *testing.T) {
	def := NewDefinition[TestEnv, struct{}]()
	l1 := def.AddLocus("L1", LocusBehavioral)
	l1.AddGene("G1", func(ctx Context[TestEnv, struct{}]) {})

	config := EngineConfig[TestEnv, struct{}]{
		PopulationSize:              10,
		MaxGenerations:              1,
		FitnessFunc: func(g Genome, env TestEnv) []float64 {
			return []float64{5.0}
		},
		ConcurrencyLimit:            4,
		DisableParallelFitness:     true,
		DisableParallelReproduction: true,
	}

	engine, err := NewEngine[TestEnv, struct{}](config)
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}

	_, err = engine.Run(def)
	if err != nil {
		t.Fatalf("Engine.Run failed with parallel disabled flags: %v", err)
	}

	if !engine.Config.DisableParallelFitness || !engine.Config.DisableParallelReproduction {
		t.Error("Disable parallel flags were not preserved in EngineConfig")
	}
}

func TestEngine_SequenceGenome(t *testing.T) {
	// 1. Test randomPermutation directly
	min, max := 1, 10
	size := 20
	pop := make(Population[TestEnv, struct{}], size)
	for i := 0; i < size; i++ {
		pop[i] = NewIndividual[TestEnv, struct{}](SequenceGenome(randomPermutation(min, max)))
	}
	if len(pop) != size {
		t.Fatalf("Expected population size %d, got %d", size, len(pop))
	}
	for i, ind := range pop {
		sg, ok := ind.Genome.(SequenceGenome)
		if !ok {
			t.Fatalf("Individual %d genome is not SequenceGenome", i)
		}
		if sg.Size() != max-min+1 {
			t.Errorf("Individual %d genome size expected %d, got %d", i, max-min+1, sg.Size())
		}
		// Check for duplicates
		seen := make(map[int]bool)
		for _, val := range sg {
			if val < min || val > max {
				t.Errorf("Value %d out of bounds [%d, %d]", val, min, max)
			}
			if seen[val] {
				t.Errorf("Duplicate value %d found in genome %v", val, sg)
			}
			seen[val] = true
		}
	}

	// 2. Test SwapMutator correctness
	mutator := SwapMutator{Probability: 1.0}
	parent := SequenceGenome{1, 2, 3, 4, 5}
	mutated := mutator.Mutate(parent).(SequenceGenome)
	if len(mutated) != len(parent) {
		t.Errorf("Mutated genome length changed: expected %d, got %d", len(parent), len(mutated))
	}
	// Check that mutated is a permutation of parent and not identical (since probability is 1.0 and size > 1)
	sameCount := 0
	parentSeen := make(map[int]bool)
	mutatedSeen := make(map[int]bool)
	for i := 0; i < len(parent); i++ {
		parentSeen[parent[i]] = true
		mutatedSeen[mutated[i]] = true
		if parent[i] == mutated[i] {
			sameCount++
		}
	}
	if len(mutatedSeen) != len(parentSeen) {
		t.Errorf("Mutated genome has duplicates or missing elements: %v", mutated)
	}
	for val := range parentSeen {
		if !mutatedSeen[val] {
			t.Errorf("Element %d from parent not found in mutated genome %v", val, mutated)
		}
	}
	if sameCount == len(parent) {
		t.Errorf("Mutated genome is identical to parent even though probability is 1.0")
	}

	// 3. Test PMXCrossover correctness
	crossover := PMXCrossover{}
	p1 := SequenceGenome{1, 2, 3, 4, 5, 6, 7, 8}
	p2 := SequenceGenome{8, 7, 6, 5, 4, 3, 2, 1}
	
	off1, off2 := crossover.Crossover(p1, p2)
	sg1, ok1 := off1.(SequenceGenome)
	sg2, ok2 := off2.(SequenceGenome)
	if !ok1 || !ok2 {
		t.Fatal("Offspring are not SequenceGenome")
	}
	if len(sg1) != len(p1) || len(sg2) != len(p2) {
		t.Errorf("Offspring lengths incorrect: got %d and %d", len(sg1), len(sg2))
	}
	
	// Verify permutation property (no duplicates)
	seen1 := make(map[int]bool)
	seen2 := make(map[int]bool)
	for i := 0; i < len(sg1); i++ {
		if seen1[sg1[i]] {
			t.Errorf("Duplicate element %d found in offspring 1: %v", sg1[i], sg1)
		}
		seen1[sg1[i]] = true
		if seen2[sg2[i]] {
			t.Errorf("Duplicate element %d found in offspring 2: %v", sg2[i], sg2)
		}
		seen2[sg2[i]] = true
	}

	// 4. Run full Engine with SequenceGenome to optimize a sequence towards target
	target := []int{1, 2, 3, 4, 5}
	fitnessFunc := func(g Genome, env TestEnv) []float64 {
		sg := g.(SequenceGenome)
		// Count correct positions (higher is better)
		score := 0.0
		for i := 0; i < len(sg); i++ {
			if sg[i] == target[i] {
				score += 1.0
			}
		}
		return []float64{score}
	}

	config := EngineConfig[TestEnv, struct{}]{
		PopulationSize: 20,
		MaxGenerations: 50,
		FitnessFunc:    fitnessFunc,
		Selector:       GenericTournamentSelector[TestEnv, struct{}]{Size: 3},
		PopulationFunc: func(def *Definition[TestEnv, struct{}], size int) Population[TestEnv, struct{}] {
			pop := make(Population[TestEnv, struct{}], size)
			for i := 0; i < size; i++ {
				pop[i] = NewIndividual[TestEnv, struct{}](SequenceGenome(randomPermutation(1, 5)))
			}
			return pop
		},
		Elitism:      2,
	}

	engine, err := NewEngine[TestEnv, struct{}](config)
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}

	best, err := engine.Run(nil) // nil definition because SequenceGenome doesn't use Categorical Loci
	if err != nil {
		t.Fatalf("Engine.Run failed: %v", err)
	}

	if best == nil {
		t.Fatal("Engine.Run returned nil best individual")
	}

	bestSG := best.Genome.(SequenceGenome)
	t.Logf("Best Sequence evolved: %v with fitness %v", bestSG, best.Fitness)
	if len(bestSG) != 5 {
		t.Errorf("Expected evolved sequence of length 5, got %d", len(bestSG))
	}
}

func TestEngine_NativeSequenceLoci(t *testing.T) {
	// Setup Categorical definition
	def := NewDefinition[TestEnv, *CounterState]()
	
	// Register a native sequence locus via Definition schema
	locus := def.AddLocus("order", LocusSequence)
	locus.AddSequenceGene("range", 1, 3)

	// behavioral gene
	lBehavior := def.AddLocus("Action", LocusBehavioral)
	lBehavior.AddGene("ExecuteSequence", func(ctx Context[TestEnv, *CounterState]) {
		seq := ctx.Individual.GetSequence("order")
		
		// Execute parameters in order defined by the sequence chromosome
		for _, idx := range seq {
			var paramName string
			switch idx {
			case 1:
				paramName = "Param1"
			case 2:
				paramName = "Param2"
			case 3:
				paramName = "Param3"
			}
			val := ctx.Individual.GetParameter(paramName)
			if val != nil {
				if v, ok := val.(int); ok {
					ctx.Individual.State.Count += v
				}
			}
		}
	})

	// three parameter genes
	lParam1 := def.AddLocus("Param1", LocusParameter)
	lParam1.AddParameterGene("P1", 10)
	
	lParam2 := def.AddLocus("Param2", LocusParameter)
	lParam2.AddParameterGene("P2", 20)
	
	lParam3 := def.AddLocus("Param3", LocusParameter)
	lParam3.AddParameterGene("P3", 30)

	// 1. Test Population Generation using DefaultPopulationFunc
	size := 10
	pop := DefaultPopulationFunc[TestEnv, *CounterState](def, size)
	if len(pop) != size {
		t.Fatalf("Expected population size %d, got %d", size, len(pop))
	}

	for _, ind := range pop {
		_, ok := ind.Genome.(CompositeGenome)
		if !ok {
			t.Fatal("Individual genome is not CompositeGenome")
		}
		
		// Check Sequence
		seq := ind.GetSequence("order")
		if len(seq) != 3 {
			t.Errorf("Expected sequence size 3, got %d", len(seq))
		}
	}

	// 2. Test Mutator and Crossover
	parent1 := pop[0].Genome.(CompositeGenome)
	parent2 := pop[1].Genome.(CompositeGenome)

	cross := DefaultCrossoverer{}
	off1, off2 := cross.Crossover(parent1, parent2)
	
	_, ok1 := off1.(CompositeGenome)
	_, ok2 := off2.(CompositeGenome)
	if !ok1 || !ok2 {
		t.Fatal("Crossover offspring are not CompositeGenome")
	}

	mut := DefaultMutator{Probability: 1.0}
	mutated := mut.Mutate(parent1).(CompositeGenome)
	if len(mutated) != 2 { // categorical and order
		t.Fatal("Mutated genome chromosomes count changed")
	}

	// 3. Test Expression using state interaction
	ind := pop[0]
	state := &CounterState{Count: 0}
	ind.State = state
	ind.Express(context.Background(), TestEnv{})
	
	// Value should be: Sum of parameters (10 + 20 + 30) = 60
	if state.Count != 60 {
		t.Errorf("Expected state.Count to be 60, got %d", state.Count)
	}
}

func TestEngine_MultipleNativeSequenceLoci(t *testing.T) {
	// Setup definition
	def := NewDefinition[TestEnv, struct{}]()
	def.AddLocus("L1", LocusParameter).AddParameterGene("P1", "val1")

	// Add multiple sequence loci
	locus1 := def.AddLocus("Collect Order", LocusSequence)
	locus1.AddSequenceGene("range", 1, 5)
	lSeq2 := def.AddLocus("Distribute Order", LocusSequence)
	lSeq2.AddSequenceGene("range", 6, 10)

	// 1. Test Population Generation
	size := 5
	pop := DefaultPopulationFunc[TestEnv, struct{}](def, size)
	if len(pop) != size {
		t.Fatalf("Expected population size %d, got %d", size, len(pop))
	}

	for i, ind := range pop {
		_, ok := ind.Genome.(CompositeGenome)
		if !ok {
			t.Fatalf("Individual %d genome is not CompositeGenome", i)
		}
		
		// Get sequences using new GetSequence API
		collectSeq := ind.GetSequence("Collect Order")
		distSeq := ind.GetSequence("Distribute Order")
		
		if len(collectSeq) != 5 {
			t.Errorf("Expected Collect Order sequence length 5, got %d", len(collectSeq))
		}
		if len(distSeq) != 5 {
			t.Errorf("Expected Distribute Order sequence length 5, got %d", len(distSeq))
		}
		
		// Verify permutation properties (no duplicates, correct ranges)
		seenCollect := make(map[int]bool)
		for _, val := range collectSeq {
			if val < 1 || val > 5 {
				t.Errorf("Collect value %d out of bounds", val)
			}
			if seenCollect[val] {
				t.Errorf("Duplicate value %d in Collect Order", val)
			}
			seenCollect[val] = true
		}
		
		seenDist := make(map[int]bool)
		for _, val := range distSeq {
			if val < 6 || val > 10 {
				t.Errorf("Distribute value %d out of bounds", val)
			}
			if seenDist[val] {
				t.Errorf("Duplicate value %d in Distribute Order", val)
			}
			seenDist[val] = true
		}
	}

	// 2. Test Mutator and Crossover
	parent1 := pop[0].Genome.(CompositeGenome)
	parent2 := pop[1].Genome.(CompositeGenome)

	cross := DefaultCrossoverer{}
	off1, off2 := cross.Crossover(parent1, parent2)
	
	cg1, ok1 := off1.(CompositeGenome)
	cg2, ok2 := off2.(CompositeGenome)
	if !ok1 || !ok2 {
		t.Fatal("Offspring are not CompositeGenome")
	}

	// Verify offspring sequence loci preserve permutation properties after crossover
	for _, cg := range []CompositeGenome{cg1, cg2} {
		collect := cg["Collect Order"].(SequenceGenome)
		dist := cg["Distribute Order"].(SequenceGenome)
		
		if len(collect) != 5 || len(dist) != 5 {
			t.Fatalf("Offspring sequence lengths incorrect")
		}
		
		// Duplicate check
		seenC := make(map[int]bool)
		for _, val := range collect {
			if seenC[val] {
				t.Errorf("Duplicate found in crossed Collect Order sequence")
			}
			seenC[val] = true
		}
		
		seenD := make(map[int]bool)
		for _, val := range dist {
			if seenD[val] {
				t.Errorf("Duplicate found in crossed Distribute Order sequence")
			}
			seenD[val] = true
		}
	}

	mut := DefaultMutator{Probability: 1.0}
	mutated := mut.Mutate(parent1).(CompositeGenome)
	
	// Verify mutated sequence loci also preserve permutation property
	mCollect := mutated["Collect Order"].(SequenceGenome)
	mDist := mutated["Distribute Order"].(SequenceGenome)
	if len(mCollect) != 5 || len(mDist) != 5 {
		t.Fatalf("Mutated sequence lengths incorrect")
	}
	seenMC := make(map[int]bool)
	for _, val := range mCollect {
		if seenMC[val] {
			t.Errorf("Duplicate found in mutated Collect Order sequence")
		}
		seenMC[val] = true
	}
}

func TestEngine_SerializationWithSequenceLoci(t *testing.T) {
	def := NewDefinition[TestEnv, struct{}]()
	def.AddLocus("L1", LocusParameter).AddParameterGene("P1", "val1")
	locus := def.AddLocus("order", LocusSequence)
	locus.AddSequenceGene("range", 1, 5)

	pop := DefaultPopulationFunc[TestEnv, struct{}](def, 1)
	cg := pop[0].Genome.(CompositeGenome)

	encoded, err := EncodeGenome(cg)
	if err != nil {
		t.Fatalf("Failed to encode: %v", err)
	}

	decoded, err := DecodeGenome[TestEnv, struct{}](def, encoded)
	if err != nil {
		t.Fatalf("Failed to decode: %v", err)
	}

	compDecoded, ok := decoded.(CompositeGenome)
	if !ok {
		t.Fatalf("Expected decoded genome to be CompositeGenome")
	}

	decodedOrder := compDecoded["order"].(SequenceGenome)
	cgOrder := cg["order"].(SequenceGenome)
	if len(decodedOrder) != 5 {
		t.Fatalf("Decoded genome sequence length incorrect: %v", decodedOrder)
	}

	for i, val := range decodedOrder {
		if val != cgOrder[i] {
			t.Errorf("Mismatch at index %d: expected %d, got %d", i, cgOrder[i], val)
		}
	}
}

