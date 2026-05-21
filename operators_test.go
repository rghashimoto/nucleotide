package nucleotide

import (
	"testing"
)

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
