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
	LocusSequence
)

// Genome represents the genetic material of an individual.
type Genome interface {
	Size() int
	Copy() Genome
}

// Context provides access to the individual's state and the environment during expression.
type Context[Env any, State any] struct {
	Ctx        context.Context
	Individual *Individual[Env, State]
	Env        Env
}

// SequencingContext provides information to the sequencer.
type SequencingContext[Env any, State any] struct {
	BehavioralLoci      []*Locus[Env, State]
	SelectedGeneIDs     []string
	SelectedGeneIndices []int
}

// Gene represents an allele at a specific locus.
type Gene[Env any, State any] struct {
	ID       string
	Callback func(ctx Context[Env, State])
	Value    interface{}
}

// Locus represents a specific position in the genome. Note: "Loci" (pronounced lo-sigh) is the plural form of "Locus".
type Locus[Env any, State any] struct {
	ID            string
	Type          LocusType
	Immutable     bool
	PossibleGenes []Gene[Env, State]
	SeqMin        int
	SeqMax        int
}

// AddGene adds a possible gene to the locus.
func (l *Locus[Env, State]) AddGene(id string, callback func(ctx Context[Env, State])) {
	l.PossibleGenes = append(l.PossibleGenes, Gene[Env, State]{ID: id, Callback: callback})
}

// AddParameterGene adds a gene that holds a value.
func (l *Locus[Env, State]) AddParameterGene(id string, value interface{}) {
	l.PossibleGenes = append(l.PossibleGenes, Gene[Env, State]{ID: id, Value: value})
}

// AddConfigGene adds a gene for framework configuration.
func (l *Locus[Env, State]) AddConfigGene(id string, value interface{}) {
	l.PossibleGenes = append(l.PossibleGenes, Gene[Env, State]{ID: id, Value: value})
}

// AddSequenceGene configures the sequence range.
func (l *Locus[Env, State]) AddSequenceGene(id string, min, max int) {
	l.SeqMin = min
	l.SeqMax = max
}

// Definition defines the structure of the genome (the set of loci). Note: "Loci" is the plural form of "Locus".
type Definition[Env any, State any] struct {
	Loci []*Locus[Env, State]
}

// NewDefinition creates a new definition with default configuration loci.
func NewDefinition[Env any, State any]() *Definition[Env, State] {
	d := &Definition[Env, State]{}
	// Add default Execution Order locus
	execLocus := d.AddLocus("Execution Order", LocusConfig)
	execLocus.Immutable = true
	execLocus.AddConfigGene("Sequential", "sequential")
	return d
}

// AddLocus adds a new locus to the definition.
func (d *Definition[Env, State]) AddLocus(id string, lType LocusType) *Locus[Env, State] {
	l := &Locus[Env, State]{ID: id, Type: lType, Immutable: false}
	if lType == LocusConfig {
		l.Immutable = true
	}
	d.Loci = append(d.Loci, l)
	return l
}

// AddSequenceGene on Definition adds a sequence gene configuration to the last added locus.
func (d *Definition[Env, State]) AddSequenceGene(id string, min, max int) {
	if len(d.Loci) > 0 {
		d.Loci[len(d.Loci)-1].AddSequenceGene(id, min, max)
	}
}

// PopulationFunc is a function that creates an initial population.
type PopulationFunc[Env any, State any] func(def *Definition[Env, State], size int) Population[Env, State]

// DefaultPopulationFunc creates a random population based on the definition.
func DefaultPopulationFunc[Env any, State any](def *Definition[Env, State], size int) Population[Env, State] {
	pop := make(Population[Env, State], size)
	for i := 0; i < size; i++ {
		indices := make([]int, len(def.Loci))
		var seqLoci []*Locus[Env, State]
		for j, locus := range def.Loci {
			if locus.Type == LocusSequence {
				seqLoci = append(seqLoci, locus)
			} else if len(locus.PossibleGenes) > 0 {
				indices[j] = rand.Intn(len(locus.PossibleGenes))
			}
		}

		catGenome := &CategoricalGenome[Env, State]{
			Definition:  def,
			GeneIndices: indices,
		}

		if len(seqLoci) > 0 {
			comp := make(CompositeGenome)
			comp["categorical"] = catGenome
			for _, locus := range seqLoci {
				comp[locus.ID] = SequenceGenome(randomPermutation(locus.SeqMin, locus.SeqMax))
			}
			pop[i] = NewIndividual[Env, State](comp)
		} else {
			pop[i] = NewIndividual[Env, State](catGenome)
		}
	}
	return pop
}

// CategoricalGenome represents a genome where each locus has a specific gene chosen.
type CategoricalGenome[Env any, State any] struct {
	Definition  *Definition[Env, State]
	GeneIndices []int
}

func (g *CategoricalGenome[Env, State]) Size() int {
	return len(g.GeneIndices)
}

func (g *CategoricalGenome[Env, State]) Copy() Genome {
	newIndices := make([]int, len(g.GeneIndices))
	copy(newIndices, g.GeneIndices)
	return &CategoricalGenome[Env, State]{
		Definition:  g.Definition,
		GeneIndices: newIndices,
	}
}

func (g *CategoricalGenome[Env, State]) GetIndices() []int {
	return g.GeneIndices
}

func (g *CategoricalGenome[Env, State]) SetIndices(indices []int) {
	g.GeneIndices = indices
}

func (g *CategoricalGenome[Env, State]) GetDefinition() interface{} {
	return g.Definition
}

func (g *CategoricalGenome[Env, State]) GetLocus(i int) (int, bool, bool) {
	if i < 0 || i >= len(g.Definition.Loci) {
		return 0, false, false
	}
	l := g.Definition.Loci[i]
	return len(l.PossibleGenes), l.Immutable, true
}

func (g *CategoricalGenome[Env, State]) GetGenePairs() []LocusGenePair {
	pairs := make([]LocusGenePair, 0, len(g.GeneIndices))
	for i, geneIdx := range g.GeneIndices {
		locus := g.Definition.Loci[i]
		if locus.Type == LocusSequence {
			continue
		}
		if geneIdx >= 0 && geneIdx < len(locus.PossibleGenes) {
			pairs = append(pairs, LocusGenePair{
				LocusID: locus.ID,
				GeneID:  locus.PossibleGenes[geneIdx].ID,
			})
		}
	}
	return pairs
}


// CompositeGenome represents a multi-chromosomal genome mapped by name.
type CompositeGenome map[string]Genome

func (g CompositeGenome) Size() int {
	size := 0
	for _, sub := range g {
		size += sub.Size()
	}
	return size
}

func (g CompositeGenome) Copy() Genome {
	newComp := make(CompositeGenome)
	for k, sub := range g {
		newComp[k] = sub.Copy()
	}
	return newComp
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

// SequenceGenome represents a permutation genome.
type SequenceGenome []int

func (g SequenceGenome) Size() int { return len(g) }

func (g SequenceGenome) Copy() Genome {
	newG := make(SequenceGenome, len(g))
	copy(newG, g)
	return newG
}

func randomPermutation(min, max int) []int {
	if min > max {
		return nil
	}
	size := max - min + 1
	perm := make([]int, size)
	for i := 0; i < size; i++ {
		perm[i] = min + i
	}
	rand.Shuffle(size, func(i, j int) {
		perm[i], perm[j] = perm[j], perm[i]
	})
	return perm
}

