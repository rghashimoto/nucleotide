package nucleotide

import (
	"math/rand"
)

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

// GaussianMutator adds Gaussian distributed noise to FloatGenomes.
type GaussianMutator struct {
	Probability float64 // mutation probability per gene
	StdDev      float64 // standard deviation of the Gaussian noise
}

func (m GaussianMutator) Mutate(g Genome) Genome {
	if fg, ok := g.(FloatGenome); ok {
		newG := fg.Copy().(FloatGenome)
		for i := range newG {
			if rand.Float64() < m.Probability {
				newG[i] += rand.NormFloat64() * m.StdDev
			}
		}
		return newG
	}
	return g
}

// CategoricalCreepMutator shifts the selected gene index of CategoricalGenomes to an adjacent option.
type CategoricalCreepMutator struct {
	Probability float64 // mutation probability per locus
}

func (m CategoricalCreepMutator) Mutate(g Genome) Genome {
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
			if !ok || immutable || numGenes <= 1 {
				continue
			}

			if rand.Float64() < m.Probability {
				currIdx := indices[i]
				var shift int
				if rand.Float64() < 0.5 {
					shift = 1
				} else {
					shift = -1
				}

				newIdx := currIdx + shift
				if newIdx < 0 {
					newIdx = 0
				}
				if newIdx >= numGenes {
					newIdx = numGenes - 1
				}
				indices[i] = newIdx
			}
		}
		newG.SetIndices(indices)
		return newG
	}
	return g
}

// DefaultMutator performs dynamic fallback mutation based on genome type.
type DefaultMutator struct {
	Probability float64
}

func (m DefaultMutator) Mutate(g Genome) Genome {
	prob := m.Probability
	if prob <= 0 {
		prob = 0.1
	}
	switch genome := g.(type) {
	case BitGenome:
		return BitFlipMutator{Probability: prob}.Mutate(genome)
	case FloatGenome:
		return GaussianMutator{Probability: prob, StdDev: 0.1}.Mutate(genome)
	default:
		return CategoricalMutator{Probability: prob}.Mutate(genome)
	}
}
