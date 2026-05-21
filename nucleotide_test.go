package nucleotide

import (
	"context"
	"math/rand"
	"os"
	"testing"
)

type TestEnv struct{}

func TestBitGenome_Size(t *testing.T) {
	g := BitGenome{true, false, true}
	if g.Size() != 3 {
		t.Errorf("Expected size 3, got %d", g.Size())
	}
}

func TestBitGenome_Copy(t *testing.T) {
	g := BitGenome{true, false, true}
	cp := g.Copy().(BitGenome)
	if len(cp) != len(g) {
		t.Errorf("Expected copy length %d, got %d", len(g), len(cp))
	}
	cp[0] = false
	if g[0] != true {
		t.Errorf("Copy should be deep, but modifying copy modified original")
	}
}

func TestFloatGenome_Copy(t *testing.T) {
	g := FloatGenome{1.0, 2.0}
	cp := g.Copy().(FloatGenome)
	if len(cp) != 2 || cp[0] != 1.0 {
		t.Error("FloatGenome copy failed")
	}
}

func TestPopulation_Methods(t *testing.T) {
	pop := Population[TestEnv, struct{}]{
		{Fitness: 10},
		{Fitness: 30},
		{Fitness: 20},
	}
	if pop.Best().Fitness != 30 {
		t.Errorf("Best() failed: got %f", pop.Best().Fitness)
	}
	if pop.AverageFitness() != 20 {
		t.Errorf("AverageFitness() failed: got %f", pop.AverageFitness())
	}
	
	empty := Population[TestEnv, struct{}]{}
	if empty.Best() != nil {
		t.Error("Best() on empty population should be nil")
	}
	if empty.AverageFitness() != 0 {
		t.Error("AverageFitness() on empty population should be 0")
	}
}

func TestGenericTournamentSelector(t *testing.T) {
	pop := Population[TestEnv, struct{}]{
		{Fitness: 10},
		{Fitness: 20},
		{Fitness: 30},
	}
	s := GenericTournamentSelector[TestEnv, struct{}]{Size: 2}
	selected := s.SelectTyped(pop)
	if selected == nil {
		t.Fatal("Selected individual is nil")
	}
	
	// Test interface implementation
	var is Selector = s
	selInterface := is.Select(pop)
	if selInterface.(*Individual[TestEnv, struct{}]) == nil {
		t.Error("Interface Select failed")
	}
}

func TestEngine_Run_Categorical(t *testing.T) {
	def := NewDefinition[TestEnv, struct{}]()
	l1 := def.AddLocus("L1", LocusBehavioral)
	l1.AddGene("G1", func(ctx Context[TestEnv, struct{}]) {})
	
	config := EngineConfig[TestEnv, struct{}]{
		PopulationSize: 10,
		MaxGenerations: 2,
		FitnessFunc: func(g Genome, env TestEnv) float64 {
			return 1.0
		},
		Selector:     GenericTournamentSelector[TestEnv, struct{}]{Size: 3},
		Crossoverers: []Crossoverer{SinglePointCrossover{}},
		Mutators:     []Mutator{CategoricalMutator{Probability: 0.1}},
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
	if best == nil {
		t.Fatal("Engine.Run returned nil best")
	}
	if engine.Generation != 2 {
		t.Errorf("Expected 2 generations, got %d", engine.Generation)
	}
}

func TestElitism_TopN(t *testing.T) {
	pop := Population[TestEnv, struct{}]{
		{Fitness: 10, Genome: BitGenome{false}},
		{Fitness: 30, Genome: BitGenome{true}},
		{Fitness: 20, Genome: BitGenome{false}},
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

func TestBitFlipMutator(t *testing.T) {
	g := BitGenome{false, false, false}
	m := BitFlipMutator{Probability: 1.0}
	mutated := m.Mutate(g).(BitGenome)
	for _, b := range mutated {
		if !b {
			t.Error("BitFlipMutator failed to flip bits")
		}
	}
}

func TestSerialization_Robust(t *testing.T) {
	def := NewDefinition[TestEnv, struct{}]()
	l1 := def.AddLocus("L1", LocusBehavioral)
	l1.AddGene("G1", func(ctx Context[TestEnv, struct{}]) {})
	l1.AddGene("G2", func(ctx Context[TestEnv, struct{}]) {})
	
	g := &CategoricalGenome[TestEnv, struct{}]{
		Definition:  def,
		GeneIndices: []int{0, 1}, // Execution Order (Sequential), L1 (G2)
	}
	
	filename := "robust_genome.json"
	defer os.Remove(filename)
	
	if err := SaveGenome(g, filename); err != nil {
		t.Fatalf("Failed to save: %v", err)
	}
	
	// Verify JSON content uses IDs
	content, _ := os.ReadFile(filename)
	if !contains(string(content), "Execution Order") || !contains(string(content), "G2") {
		t.Errorf("JSON does not contain expected IDs: %s", string(content))
	}
	
	// Load using a DIFFERENT definition with SAME IDs but DIFFERENT order
	def2 := NewDefinition[TestEnv, struct{}]()
	l1_2 := def2.AddLocus("L1", LocusBehavioral)
	l1_2.AddGene("G2", func(ctx Context[TestEnv, struct{}]) {}) // G2 is now index 0
	l1_2.AddGene("G1", func(ctx Context[TestEnv, struct{}]) {})
	
	loaded, err := LoadGenome(def2, filename)
	if err != nil {
		t.Fatalf("Failed to load: %v", err)
	}
	
	if loaded.GeneIndices[1] != 0 {
		t.Errorf("LoadGenome failed to map IDs correctly: expected index 0 (G2), got %d", loaded.GeneIndices[1])
	}
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && stringContains(s, substr)
}

func stringContains(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
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

func TestMemorySerialization(t *testing.T) {
	def := NewDefinition[TestEnv, struct{}]()
	l1 := def.AddLocus("L1", LocusBehavioral)
	l1.AddGene("G1", func(ctx Context[TestEnv, struct{}]) {})
	
	g := &CategoricalGenome[TestEnv, struct{}]{
		Definition:  def,
		GeneIndices: []int{0, 0},
	}
	
	bytes, err := EncodeGenome(g)
	if err != nil {
		t.Fatalf("Encode failed: %v", err)
	}
	
	loaded, err := DecodeGenome(def, bytes)
	if err != nil {
		t.Fatalf("Decode failed: %v", err)
	}
	
	if loaded.GeneIndices[1] != g.GeneIndices[1] {
		t.Errorf("Mismatch after decode")
	}
}

type CounterState struct {
	Count int
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

func TestTwoPointCrossover(t *testing.T) {
	// Test on BitGenome
	p1 := BitGenome{true, true, true, true, true}
	p2 := BitGenome{false, false, false, false, false}
	c := TwoPointCrossover{}
	
	off1, off2 := c.Crossover(p1, p2)
	bg1 := off1.(BitGenome)
	bg2 := off2.(BitGenome)
	
	if len(bg1) != 5 || len(bg2) != 5 {
		t.Errorf("Expected offspring size 5, got %d and %d", len(bg1), len(bg2))
	}
	
	// Test on CategoricalGenome
	def := NewDefinition[TestEnv, struct{}]()
	l1 := def.AddLocus("L1", LocusBehavioral)
	l1.AddGene("G1", func(ctx Context[TestEnv, struct{}]) {})
	l2 := def.AddLocus("L2", LocusBehavioral)
	l2.AddGene("G1", func(ctx Context[TestEnv, struct{}]) {})
	l3 := def.AddLocus("L3", LocusBehavioral)
	l3.AddGene("G1", func(ctx Context[TestEnv, struct{}]) {})
	
	cg1 := &CategoricalGenome[TestEnv, struct{}]{
		Definition:  def,
		GeneIndices: []int{0, 0, 0, 0},
	}
	cg2 := &CategoricalGenome[TestEnv, struct{}]{
		Definition:  def,
		GeneIndices: []int{0, 1, 1, 1},
	}
	
	o1, o2 := c.Crossover(cg1, cg2)
	cog1 := o1.(*CategoricalGenome[TestEnv, struct{}])
	cog2 := o2.(*CategoricalGenome[TestEnv, struct{}])
	
	if len(cog1.GeneIndices) != 4 || len(cog2.GeneIndices) != 4 {
		t.Error("TwoPointCrossover on CategoricalGenome failed")
	}
}

func TestUniformCrossover(t *testing.T) {
	p1 := BitGenome{true, true, true, true, true}
	p2 := BitGenome{false, false, false, false, false}
	c := UniformCrossover{Probability: 0.5}
	
	off1, off2 := c.Crossover(p1, p2)
	bg1 := off1.(BitGenome)
	bg2 := off2.(BitGenome)
	
	if len(bg1) != 5 || len(bg2) != 5 {
		t.Error("UniformCrossover failed")
	}
}

func TestArithmeticCrossover(t *testing.T) {
	p1 := FloatGenome{1.0, 2.0, 3.0}
	p2 := FloatGenome{3.0, 4.0, 5.0}
	c := ArithmeticCrossover{Alpha: 0.5}
	
	off1, off2 := c.Crossover(p1, p2)
	fg1 := off1.(FloatGenome)
	fg2 := off2.(FloatGenome)
	
	// Value should be 0.5 * 1.0 + 0.5 * 3.0 = 2.0
	if fg1[0] != 2.0 || fg2[0] != 2.0 {
		t.Errorf("ArithmeticCrossover failed: expected 2.0, got %f and %f", fg1[0], fg2[0])
	}
}

func TestGaussianMutator(t *testing.T) {
	g := FloatGenome{0.0, 0.0, 0.0}
	m := GaussianMutator{Probability: 1.0, StdDev: 1.0}
	mutated := m.Mutate(g).(FloatGenome)
	
	allZero := true
	for _, val := range mutated {
		if val != 0.0 {
			allZero = false
		}
	}
	if allZero {
		t.Error("GaussianMutator failed to mutate values")
	}
}

func TestCategoricalCreepMutator(t *testing.T) {
	def := NewDefinition[TestEnv, struct{}]()
	l1 := def.AddLocus("L1", LocusBehavioral)
	l1.AddGene("G1", func(ctx Context[TestEnv, struct{}]) {})
	l1.AddGene("G2", func(ctx Context[TestEnv, struct{}]) {})
	l1.AddGene("G3", func(ctx Context[TestEnv, struct{}]) {})
	
	g := &CategoricalGenome[TestEnv, struct{}]{
		Definition:  def,
		GeneIndices: []int{0, 1}, // index 1 is locus L1 with current gene G2 (idx 1)
	}
	
	m := CategoricalCreepMutator{Probability: 1.0}
	mutated := m.Mutate(g).(*CategoricalGenome[TestEnv, struct{}])
	
	// The mutated index at index 1 should be either 0 (G1) or 2 (G3)
	mutIdx := mutated.GeneIndices[1]
	if mutIdx != 0 && mutIdx != 2 {
		t.Errorf("CategoricalCreepMutator failed: expected index 0 or 2, got %d", mutIdx)
	}
}

func TestProbabilisticTournamentSelector(t *testing.T) {
	rand.Seed(1) // Seed for deterministic tests
	pop := Population[TestEnv, struct{}]{
		{Fitness: 10.0, Genome: BitGenome{true}},
		{Fitness: 5.0, Genome: BitGenome{true}},
		{Fitness: 1.0, Genome: BitGenome{true}},
	}

	// We use Unique: true to ensure all 3 individuals are in the tournament,
	// allowing us to isolate probabilistic behavior directly.
	sHigh := NewProbabilisticTournamentSelector[TestEnv, struct{}](3, 0.999)
	sHigh.Unique = true

	sLow := NewProbabilisticTournamentSelector[TestEnv, struct{}](3, 0.001)
	sLow.Unique = true

	bestWins := 0
	for i := 0; i < 100; i++ {
		sel := sHigh.SelectTyped(pop)
		if sel.Fitness == 10.0 {
			bestWins++
		}
	}
	if bestWins < 90 {
		t.Errorf("Expected high fitness individual to win most of the time with P=0.999, got %d wins", bestWins)
	}

	worstWins := 0
	for i := 0; i < 100; i++ {
		sel := sLow.SelectTyped(pop)
		if sel.Fitness == 1.0 {
			worstWins++
		}
	}
	if worstWins < 90 {
		t.Errorf("Expected lowest fitness individual to win most of the time with P=0.001, got %d wins", worstWins)
	}
}

func TestAdaptiveTournamentSelector(t *testing.T) {
	pop := Population[TestEnv, struct{}]{
		{Fitness: 1.0, Genome: BitGenome{true}},
		{Fitness: 2.0, Genome: BitGenome{true}},
		{Fitness: 3.0, Genome: BitGenome{true}},
		{Fitness: 4.0, Genome: BitGenome{true}},
		{Fitness: 5.0, Genome: BitGenome{true}},
	}

	progress := 0.0
	progressFunc := func() float64 {
		return progress
	}

	s := NewAdaptiveTournamentSelector[TestEnv, struct{}](1, 5, progressFunc)
	s.Unique = true // Ensure deterministic selection when size matches population size

	// At start progress = 0.0, effective size is 1.
	// Best (5.0) should not win 100% of the time.
	progress = 0.0
	fiveWinsStart := 0
	for i := 0; i < 100; i++ {
		if s.SelectTyped(pop).Fitness == 5.0 {
			fiveWinsStart++
		}
	}
	if fiveWinsStart == 100 {
		t.Errorf("Size should be 1 at start progress, but got 100%% wins for the best individual")
	}

	// At end progress = 1.0, effective size is 5.
	// Best (5.0) must win 100% of the time.
	progress = 1.0
	fiveWinsEnd := 0
	for i := 0; i < 100; i++ {
		if s.SelectTyped(pop).Fitness == 5.0 {
			fiveWinsEnd++
		}
	}
	if fiveWinsEnd != 100 {
		t.Errorf("Expected best individual to win 100%% of the time at end progress (size 5), got %d wins", fiveWinsEnd)
	}
}

func TestNichingTournamentSelector(t *testing.T) {
	// Two identical individuals (BitGenome{true, true}) with fitness 10.0
	// One different individual (BitGenome{false, false}) with fitness 8.0
	pop := Population[TestEnv, struct{}]{
		{Fitness: 10.0, Genome: BitGenome{true, true}},
		{Fitness: 10.0, Genome: BitGenome{true, true}},
		{Fitness: 8.0, Genome: BitGenome{false, false}},
	}

	// Without niching, the best (10.0) always wins
	sNoNiche := GenericTournamentSelector[TestEnv, struct{}]{Size: 3}
	sNoNiche.Unique = true
	if sNoNiche.SelectTyped(pop).Fitness != 10.0 {
		t.Errorf("Expected 10.0 without niching")
	}

	// With niching (SigmaShare = 1.0)
	// Hamming distance between identical is 0.0, which is < SigmaShare (1.0).
	// They share and penalize each other: NicheCount = 1 + 1 = 2. AdjustedFit = 10.0 / 2.0 = 5.0.
	// The different individual has distance 1.0 from others, so it doesn't share.
	// AdjustedFit of different = 8.0 / 1.0 = 8.0.
	// Since 8.0 > 5.0, the different individual wins.
	sNiche := NewNichingTournamentSelector[TestEnv, struct{}](3, 1.0, nil)
	sNiche.Unique = true
	sel := sNiche.SelectTyped(pop)
	if sel.Fitness != 8.0 {
		t.Errorf("Expected different individual with fitness 8.0 to win due to niche penalty of duplicate individuals, got %f", sel.Fitness)
	}
}

func TestUniqueTournamentSelector(t *testing.T) {
	pop := Population[TestEnv, struct{}]{
		{Fitness: 10.0, Genome: BitGenome{true}},
		{Fitness: 5.0, Genome: BitGenome{true}},
	}

	// With replacement (size 2), worst individual (5.0) can occasionally win
	sWithRep := GenericTournamentSelector[TestEnv, struct{}]{Size: 2}
	worstWinsWithRep := 0
	for i := 0; i < 100; i++ {
		if sWithRep.SelectTyped(pop).Fitness == 5.0 {
			worstWinsWithRep++
		}
	}

	// Without replacement (size 2), worst individual can NEVER win (since best is always present in a size-2 tournament of a size-2 population)
	sUnique := NewUniqueTournamentSelector[TestEnv, struct{}](2)
	worstWinsUnique := 0
	for i := 0; i < 100; i++ {
		if sUnique.SelectTyped(pop).Fitness == 5.0 {
			worstWinsUnique++
		}
	}

	if worstWinsUnique != 0 {
		t.Errorf("Expected 0 wins for worst individual in unique tournament of size 2, got %d wins", worstWinsUnique)
	}
	
	t.Logf("Worst individual won %d times with replacement, and %d times without replacement", worstWinsWithRep, worstWinsUnique)
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
		FitnessFunc: func(g Genome, env TestEnv) float64 {
			return 1.0
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
		if _, ok := engine.Config.Crossoverers[0].(DefaultCrossoverer); !ok {
			t.Errorf("Expected default crossoverer to be DefaultCrossoverer, got %T", engine.Config.Crossoverers[0])
		}
	}

	if len(engine.Config.Mutators) != 1 {
		t.Errorf("Expected 1 default mutator, got %d", len(engine.Config.Mutators))
	} else {
		if _, ok := engine.Config.Mutators[0].(DefaultMutator); !ok {
			t.Errorf("Expected default mutator to be DefaultMutator, got %T", engine.Config.Mutators[0])
		}
	}
}

type MockCrossoverer struct {
	id int
}

func (m MockCrossoverer) Crossover(p1, p2 Genome) (Genome, Genome) {
	return p1.Copy(), p2.Copy()
}

type MockMutator struct {
	id int
}

func (m MockMutator) Mutate(g Genome) Genome {
	return g.Copy()
}

func TestEngine_RoundRobinOperators(t *testing.T) {
	config := EngineConfig[TestEnv, struct{}]{
		PopulationSize: 10,
		MaxGenerations: 2,
		FitnessFunc: func(g Genome, env TestEnv) float64 {
			return 1.0
		},
		Crossoverers: []Crossoverer{
			MockCrossoverer{id: 1},
			MockCrossoverer{id: 2},
		},
		Mutators: []Mutator{
			MockMutator{id: 1},
			MockMutator{id: 2},
			MockMutator{id: 3},
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
		FitnessFunc: func(g Genome, env TestEnv) float64 {
			return 1.0
		},
		Crossoverers: []Crossoverer{
			MockCrossoverer{id: 1},
			MockCrossoverer{id: 2},
		},
		CrossovererWeights: []float64{0.0, 1.0},
		Mutators: []Mutator{
			MockMutator{id: 1},
			MockMutator{id: 2},
		},
		MutatorWeights: []float64{1.0, 0.0},
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

	// Test invalid weights validation inside NewEngine
	invalidConfig1 := config
	invalidConfig1.CrossovererWeights = []float64{1.0} // size mismatch
	_, err = NewEngine[TestEnv, struct{}](invalidConfig1)
	if err == nil {
		t.Error("Expected error for CrossovererWeights size mismatch, got nil")
	}

	invalidConfig2 := config
	invalidConfig2.CrossovererWeights = []float64{-1.0, 2.0} // negative value
	_, err = NewEngine[TestEnv, struct{}](invalidConfig2)
	if err == nil {
		t.Error("Expected error for negative CrossovererWeights, got nil")
	}

	invalidConfig3 := config
	invalidConfig3.CrossovererWeights = []float64{0.0, 0.0} // zero sum
	_, err = NewEngine[TestEnv, struct{}](invalidConfig3)
	if err == nil {
		t.Error("Expected error for zero-sum CrossovererWeights, got nil")
	}
}

