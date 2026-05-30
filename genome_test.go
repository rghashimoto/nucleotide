package nucleotide

import (
	"os"
	"testing"
)

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
		{Fitness: []float64{10}},
		{Fitness: []float64{30}},
		{Fitness: []float64{20}},
	}
	if len(pop.Best().Fitness) == 0 || pop.Best().Fitness[0] != 30 {
		t.Errorf("Best() failed: got %v", pop.Best().Fitness)
	}
	avg := pop.AverageFitness()
	if len(avg) == 0 || avg[0] != 20 {
		t.Errorf("AverageFitness() failed: got %v", avg)
	}
	
	empty := Population[TestEnv, struct{}]{}
	if empty.Best() != nil {
		t.Error("Best() on empty population should be nil")
	}
	if empty.AverageFitness() != nil {
		t.Error("AverageFitness() on empty population should be nil")
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
	
	cgLoaded := loaded.(*CategoricalGenome[TestEnv, struct{}])
	if cgLoaded.GeneIndices[1] != 0 {
		t.Errorf("LoadGenome failed to map IDs correctly: expected index 0 (G2), got %d", cgLoaded.GeneIndices[1])
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
	
	cgDecoded := loaded.(*CategoricalGenome[TestEnv, struct{}])
	if cgDecoded.GeneIndices[1] != g.GeneIndices[1] {
		t.Errorf("Mismatch after decode")
	}
}

func TestSerialization_AllGenomeTypes(t *testing.T) {
	// 1. BitGenome
	bitG := BitGenome{true, false, true}
	bitBytes, err := EncodeGenome(bitG)
	if err != nil {
		t.Fatalf("BitGenome encode failed: %v", err)
	}
	bitDec, err := DecodeGenome[TestEnv, struct{}](nil, bitBytes)
	if err != nil {
		t.Fatalf("BitGenome decode failed: %v", err)
	}
	bitDecoded, ok := bitDec.(BitGenome)
	if !ok {
		t.Fatalf("Expected BitGenome, got %T", bitDec)
	}
	if len(bitDecoded) != 3 || bitDecoded[0] != true || bitDecoded[1] != false || bitDecoded[2] != true {
		t.Errorf("BitGenome decoded content mismatch: %v", bitDecoded)
	}

	// 2. FloatGenome
	floatG := FloatGenome{1.5, -2.4, 0.0}
	floatBytes, err := EncodeGenome(floatG)
	if err != nil {
		t.Fatalf("FloatGenome encode failed: %v", err)
	}
	floatDec, err := DecodeGenome[TestEnv, struct{}](nil, floatBytes)
	if err != nil {
		t.Fatalf("FloatGenome decode failed: %v", err)
	}
	floatDecoded, ok := floatDec.(FloatGenome)
	if !ok {
		t.Fatalf("Expected FloatGenome, got %T", floatDec)
	}
	if len(floatDecoded) != 3 || floatDecoded[0] != 1.5 || floatDecoded[1] != -2.4 || floatDecoded[2] != 0.0 {
		t.Errorf("FloatGenome decoded content mismatch: %v", floatDecoded)
	}

	// 3. Standalone SequenceGenome
	seqG := SequenceGenome{4, 2, 1, 3, 5}
	seqBytes, err := EncodeGenome(seqG)
	if err != nil {
		t.Fatalf("SequenceGenome encode failed: %v", err)
	}
	seqDec, err := DecodeGenome[TestEnv, struct{}](nil, seqBytes)
	if err != nil {
		t.Fatalf("SequenceGenome decode failed: %v", err)
	}
	seqDecoded, ok := seqDec.(SequenceGenome)
	if !ok {
		t.Fatalf("Expected SequenceGenome, got %T", seqDec)
	}
	if len(seqDecoded) != 5 || seqDecoded[0] != 4 || seqDecoded[1] != 2 || seqDecoded[2] != 1 || seqDecoded[3] != 3 || seqDecoded[4] != 5 {
		t.Errorf("SequenceGenome decoded content mismatch: %v", seqDecoded)
	}
}
