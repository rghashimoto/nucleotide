package nucleotide

import (
	"math/rand"
)

// Selector defines the interface for selecting individuals from a population.
type Selector interface {
	// We use any here because Selector might work with different Individual types.
	// However, usually we want it to be specific.
	// Since Selector is an interface, and Go doesn't support generic methods in interfaces,
	// we have a few options. One is to make Selector generic too.
	Select(pop interface{}) interface{}
}

// Crossoverer defines the interface for combining two parents into offspring.
type Crossoverer interface {
	Crossover(p1, p2 Genome) (Genome, Genome)
}

// Mutator defines the interface for introducing random changes to a genome.
type Mutator interface {
	Mutate(g Genome) Genome
}

// TournamentSelector selects the best individual from a random subset.
type TournamentSelector struct {
	Size int
}

func (s TournamentSelector) Select(pop interface{}) interface{} {
	// Use type switch to handle different Population types
	// This is a bit clunky due to Go's interface/generics limitations,
	// but it allows the Engine to use the selector.
	
	// A better way is to make TournamentSelector generic if we know the type E.
	return nil // Placeholder, will fix below with a generic-friendly approach
}

// GenericTournamentSelector is a type-safe selector for a specific environment E and state S.
type GenericTournamentSelector[E any, S any] struct {
	Size int
}

func (s GenericTournamentSelector[E, S]) Select(pop interface{}) interface{} {
	return s.SelectTyped(pop.(Population[E, S]))
}

func (s GenericTournamentSelector[E, S]) SelectTyped(pop Population[E, S]) *Individual[E, S] {
	if len(pop) == 0 {
		return nil
	}
	tournament := make(Population[E, S], s.Size)
	for i := 0; i < s.Size; i++ {
		tournament[i] = pop[rand.Intn(len(pop))]
	}
	return tournament.Best()
}

// SinglePointCrossover performs crossover at a single random point.
type SinglePointCrossover struct{}

func (c SinglePointCrossover) Crossover(p1, p2 Genome) (Genome, Genome) {
	size := p1.Size()
	if size <= 1 {
		return p1.Copy(), p2.Copy()
	}

	point := rand.Intn(size-1) + 1

	// We need to handle CategoricalGenome[E any, S any]. Since we don't know E and S,
	// we can try to use type assertions for common types or reflect.
	// But wait, CategoricalGenome stores indices! The indices are not generic!
	// Only the Definition is generic.
	
	// Let's use a trick: check for a generic-independent interface if possible,
	// or just handle it if we know the structure.
	
	// Actually, we can use an interface for genomes that expose their internal indices.
	type indexedGenome interface {
		Genome
		GetIndices() []int
		SetIndices([]int)
		GetDefinition() interface{}
	}

	if g1, ok := p1.(indexedGenome); ok {
		g2 := p2.(indexedGenome)
		indices1 := g1.GetIndices()
		indices2 := g2.GetIndices()
		
		off1Indices := make([]int, size)
		off2Indices := make([]int, size)
		copy(off1Indices[:point], indices1[:point])
		copy(off1Indices[point:], indices2[point:])
		copy(off2Indices[:point], indices2[:point])
		copy(off2Indices[point:], indices1[point:])
		
		off1 := p1.Copy().(indexedGenome)
		off1.SetIndices(off1Indices)
		off2 := p2.Copy().(indexedGenome)
		off2.SetIndices(off2Indices)
		return off1, off2
	}

	switch g1 := p1.(type) {
	case BitGenome:
		g2 := p2.(BitGenome)
		off1 := make(BitGenome, size)
		off2 := make(BitGenome, size)
		copy(off1[:point], g1[:point])
		copy(off1[point:], g2[point:])
		copy(off2[:point], g2[:point])
		copy(off2[point:], g1[point:])
		return off1, off2
	}

	return p1.Copy(), p2.Copy()
}

// CategoricalMutator chooses a random gene from the possible genes for a locus.
type CategoricalMutator struct {
	Probability float64
}

func (m CategoricalMutator) Mutate(g Genome) Genome {
	// Again, we need to handle the generic CategoricalGenome[E, S].
	// Let's use the same indexedGenome interface.
	
	type mutableIndexedGenome interface {
		Genome
		GetIndices() []int
		SetIndices([]int)
		GetLocus(int) (int, bool, bool) // returns numGenes, immutable, ok
	}

	if mg, ok := g.(mutableIndexedGenome); ok {
		newG := mg.Copy().(mutableIndexedGenome)
		indices := newG.GetIndices()
		for i := range indices {
			numGenes, immutable, ok := mg.GetLocus(i)
			if !ok || immutable {
				continue
			}
			if rand.Float64() < m.Probability {
				if numGenes > 0 {
					indices[i] = rand.Intn(numGenes)
				}
			}
		}
		newG.SetIndices(indices)
		return newG
	}
	return g
}

// BitFlipMutator flips a bit with a given probability.
type BitFlipMutator struct {
	Probability float64
}

func (m BitFlipMutator) Mutate(g Genome) Genome {
	if bg, ok := g.(BitGenome); ok {
		newG := bg.Copy().(BitGenome)
		for i := range newG {
			if rand.Float64() < m.Probability {
				newG[i] = !newG[i]
			}
		}
		return newG
	}
	return g
}
