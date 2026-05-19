package nucleotide

import (
	"context"
	"io/ioutil"
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
	pop := Population[TestEnv]{
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
	
	empty := Population[TestEnv]{}
	if empty.Best() != nil {
		t.Error("Best() on empty population should be nil")
	}
	if empty.AverageFitness() != 0 {
		t.Error("AverageFitness() on empty population should be 0")
	}
}

func TestGenericTournamentSelector(t *testing.T) {
	pop := Population[TestEnv]{
		{Fitness: 10},
		{Fitness: 20},
		{Fitness: 30},
	}
	s := GenericTournamentSelector[TestEnv]{Size: 2}
	selected := s.SelectTyped(pop)
	if selected == nil {
		t.Fatal("Selected individual is nil")
	}
	
	// Test interface implementation
	var is Selector = s
	selInterface := is.Select(pop)
	if selInterface.(*Individual[TestEnv]) == nil {
		t.Error("Interface Select failed")
	}
}

func TestEngine_Run_Categorical(t *testing.T) {
	def := NewDefinition[TestEnv]()
	l1 := def.AddLocus("L1", LocusBehavioral)
	l1.AddGene("G1", func(ctx Context[TestEnv]) {})
	
	config := EngineConfig[TestEnv]{
		PopulationSize: 10,
		MaxGenerations: 2,
		FitnessFunc: func(g Genome, env TestEnv) float64 {
			return 1.0
		},
		Selector:    GenericTournamentSelector[TestEnv]{Size: 3},
		Crossoverer: SinglePointCrossover{},
		Mutator:     CategoricalMutator{Probability: 0.1},
		Elitism:     1,
	}
	engine := NewEngine[TestEnv](config)
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
	pop := Population[TestEnv]{
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
	def := NewDefinition[TestEnv]()
	l1 := def.AddLocus("L1", LocusBehavioral)
	l1.AddGene("G1", func(ctx Context[TestEnv]) {})
	l1.AddGene("G2", func(ctx Context[TestEnv]) {})
	
	g := &CategoricalGenome[TestEnv]{
		Definition:  def,
		GeneIndices: []int{0, 1}, // Execution Order (Sequential), L1 (G2)
	}
	
	filename := "robust_genome.json"
	defer os.Remove(filename)
	
	if err := SaveGenome(g, filename); err != nil {
		t.Fatalf("Failed to save: %v", err)
	}
	
	// Verify JSON content uses IDs
	content, _ := ioutil.ReadFile(filename)
	if !contains(string(content), "Execution Order") || !contains(string(content), "G2") {
		t.Errorf("JSON does not contain expected IDs: %s", string(content))
	}
	
	// Load using a DIFFERENT definition with SAME IDs but DIFFERENT order
	def2 := NewDefinition[TestEnv]()
	l1_2 := def2.AddLocus("L1", LocusBehavioral)
	l1_2.AddGene("G2", func(ctx Context[TestEnv]) {}) // G2 is now index 0
	l1_2.AddGene("G1", func(ctx Context[TestEnv]) {})
	
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
	def := NewDefinition[TestEnv]()
	l1 := def.AddLocus("L1", LocusBehavioral)
	count := 0
	l1.AddGene("G1", func(ctx Context[TestEnv]) { count++ })
	l1.AddGene("G2", func(ctx Context[TestEnv]) { count++ })
	
	g := &CategoricalGenome[TestEnv]{
		Definition:  def,
		GeneIndices: []int{0, 0, 1}, // Seq, G1, G2
	}
	ind := NewIndividual[TestEnv](g)
	
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately
	
	ind.Express(ctx, TestEnv{})
	if count > 0 {
		t.Errorf("Express did not respect cancellation: executed %d genes", count)
	}
}
