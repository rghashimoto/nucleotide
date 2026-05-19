package nucleotide

import (
	"context"
	"math/rand"
)

// LocusType defines the purpose of a locus.
type LocusType int

const (
	LocusBehavioral LocusType = iota
	LocusParameter
	LocusConfig
)

// Genome represents the genetic material of an individual.
type Genome interface {
	Size() int
	Copy() Genome
}

// Context provides access to the individual's state and the environment during expression.
type Context[E any] struct {
	Ctx        context.Context
	Individual *Individual[E]
	Env        E
}

// SequencingContext provides information to the sequencer.
type SequencingContext[E any] struct {
	BehavioralLoci      []*Locus[E]
	SelectedGeneIDs     []string
	SelectedGeneIndices []int
}

// Gene represents an allele at a specific locus.
type Gene[E any] struct {
	ID       string
	Callback func(ctx Context[E])
	Value    interface{}
}

// Locus represents a specific position in the genome.
type Locus[E any] struct {
	ID            string
	Type          LocusType
	Immutable     bool
	PossibleGenes []Gene[E]
}

// AddGene adds a possible gene to the locus.
func (l *Locus[E]) AddGene(id string, callback func(ctx Context[E])) {
	l.PossibleGenes = append(l.PossibleGenes, Gene[E]{ID: id, Callback: callback})
}

// AddParameterGene adds a gene that holds a value.
func (l *Locus[E]) AddParameterGene(id string, value interface{}) {
	l.PossibleGenes = append(l.PossibleGenes, Gene[E]{ID: id, Value: value})
}

// AddConfigGene adds a gene for framework configuration.
func (l *Locus[E]) AddConfigGene(id string, value interface{}) {
	l.PossibleGenes = append(l.PossibleGenes, Gene[E]{ID: id, Value: value})
}

// Definition defines the structure of the genome (the set of loci).
type Definition[E any] struct {
	Loci []*Locus[E]
}

// NewDefinition creates a new definition with default configuration loci.
func NewDefinition[E any]() *Definition[E] {
	d := &Definition[E]{}
	// Add default Execution Order locus
	execLocus := d.AddLocus("Execution Order", LocusConfig)
	execLocus.Immutable = true
	execLocus.AddConfigGene("Sequential", "sequential")
	return d
}

// AddLocus adds a new locus to the definition.
func (d *Definition[E]) AddLocus(id string, lType LocusType) *Locus[E] {
	l := &Locus[E]{ID: id, Type: lType, Immutable: false}
	if lType == LocusConfig {
		l.Immutable = true
	}
	d.Loci = append(d.Loci, l)
	return l
}

// PopulationFunc is a function that creates an initial population.
type PopulationFunc[E any] func(def *Definition[E], size int) Population[E]

// DefaultPopulationFunc creates a random population based on the definition.
func DefaultPopulationFunc[E any](def *Definition[E], size int) Population[E] {
	pop := make(Population[E], size)
	for i := 0; i < size; i++ {
		indices := make([]int, len(def.Loci))
		for j, locus := range def.Loci {
			if len(locus.PossibleGenes) > 0 {
				indices[j] = rand.Intn(len(locus.PossibleGenes))
			}
		}
		pop[i] = NewIndividual[E](&CategoricalGenome[E]{
			Definition:  def,
			GeneIndices: indices,
		})
	}
	return pop
}

// CategoricalGenome represents a genome where each locus has a specific gene chosen.
type CategoricalGenome[E any] struct {
	Definition  *Definition[E]
	GeneIndices []int
}

func (g *CategoricalGenome[E]) Size() int {
	return len(g.GeneIndices)
}

func (g *CategoricalGenome[E]) Copy() Genome {
	newIndices := make([]int, len(g.GeneIndices))
	copy(newIndices, g.GeneIndices)
	return &CategoricalGenome[E]{
		Definition:  g.Definition,
		GeneIndices: newIndices,
	}
}

func (g *CategoricalGenome[E]) GetIndices() []int {
	return g.GeneIndices
}

func (g *CategoricalGenome[E]) SetIndices(indices []int) {
	g.GeneIndices = indices
}

func (g *CategoricalGenome[E]) GetDefinition() interface{} {
	return g.Definition
}

func (g *CategoricalGenome[E]) GetLocus(i int) (int, bool, bool) {
	if i < 0 || i >= len(g.Definition.Loci) {
		return 0, false, false
	}
	l := g.Definition.Loci[i]
	return len(l.PossibleGenes), l.Immutable, true
}

// BitGenome and FloatGenome are kept for compatibility.
type BitGenome []bool
func (g BitGenome) Size() int { return len(g) }
func (g BitGenome) Copy() Genome {
	newG := make(BitGenome, len(g))
	copy(newG, g)
	return newG
}

type FloatGenome []float64
func (g FloatGenome) Size() int { return len(g) }
func (g FloatGenome) Copy() Genome {
	newG := make(FloatGenome, len(g))
	copy(newG, g)
	return newG
}
