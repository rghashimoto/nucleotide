package nucleotide

import (
	"math/rand"
)

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

// TwoPointCrossover performs crossover at two random points.
type TwoPointCrossover struct{}

func (c TwoPointCrossover) Crossover(p1, p2 Genome) (Genome, Genome) {
	size := p1.Size()
	if size <= 2 {
		// Fallback to single point if size is too small
		return SinglePointCrossover{}.Crossover(p1, p2)
	}

	point1 := rand.Intn(size - 1)
	point2 := rand.Intn(size-1-point1) + point1 + 1

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
		copy(off1Indices, indices1)
		copy(off2Indices, indices2)

		// Swap the segment [point1:point2]
		copy(off1Indices[point1:point2], indices2[point1:point2])
		copy(off2Indices[point1:point2], indices1[point1:point2])

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
		copy(off1, g1)
		copy(off2, g2)

		copy(off1[point1:point2], g2[point1:point2])
		copy(off2[point1:point2], g1[point1:point2])
		return off1, off2
	}

	return p1.Copy(), p2.Copy()
}

// UniformCrossover swaps genes at each locus with a given probability.
type UniformCrossover struct {
	Probability float64 // typically 0.5
}

func (c UniformCrossover) Crossover(p1, p2 Genome) (Genome, Genome) {
	size := p1.Size()
	if size <= 0 {
		return p1.Copy(), p2.Copy()
	}

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

		for i := 0; i < size; i++ {
			if rand.Float64() < c.Probability {
				off1Indices[i] = indices2[i]
				off2Indices[i] = indices1[i]
			} else {
				off1Indices[i] = indices1[i]
				off2Indices[i] = indices2[i]
			}
		}

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

		for i := 0; i < size; i++ {
			if rand.Float64() < c.Probability {
				off1[i] = g2[i]
				off2[i] = g1[i]
			} else {
				off1[i] = g1[i]
				off2[i] = g2[i]
			}
		}
		return off1, off2
	}

	return p1.Copy(), p2.Copy()
}

// ArithmeticCrossover performs arithmetic combination of two FloatGenomes.
type ArithmeticCrossover struct {
	Alpha float64 // weight factor in [0, 1]
}

func (c ArithmeticCrossover) Crossover(p1, p2 Genome) (Genome, Genome) {
	size := p1.Size()
	if size <= 0 {
		return p1.Copy(), p2.Copy()
	}

	switch g1 := p1.(type) {
	case FloatGenome:
		g2 := p2.(FloatGenome)
		off1 := make(FloatGenome, size)
		off2 := make(FloatGenome, size)

		for i := 0; i < size; i++ {
			off1[i] = c.Alpha*g1[i] + (1.0-c.Alpha)*g2[i]
			off2[i] = c.Alpha*g2[i] + (1.0-c.Alpha)*g1[i]
		}
		return off1, off2
	}

	return p1.Copy(), p2.Copy()
}

// DefaultCrossoverer performs dynamic fallback crossover based on genome type.
type DefaultCrossoverer struct{}

func (c DefaultCrossoverer) Crossover(p1, p2 Genome) (Genome, Genome) {
	switch p1.(type) {
	case FloatGenome:
		return ArithmeticCrossover{Alpha: 0.5}.Crossover(p1, p2)
	default:
		return SinglePointCrossover{}.Crossover(p1, p2)
	}
}
