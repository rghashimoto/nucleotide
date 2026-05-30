package nucleotide

import (
	"math/rand"
)

// SinglePointCrossover performs crossover at a single random point.
type SinglePointCrossover struct{}

func (c SinglePointCrossover) Crossover(p1, p2 Genome) (Genome, Genome) {
	if comp1, ok := p1.(CompositeGenome); ok {
		comp2 := p2.(CompositeGenome)
		off1 := make(CompositeGenome)
		off2 := make(CompositeGenome)
		for k, sub1 := range comp1 {
			sub2 := comp2[k]
			if _, isSeq := sub1.(SequenceGenome); isSeq {
				o1, o2 := PMXCrossover{}.Crossover(sub1, sub2)
				off1[k] = o1
				off2[k] = o2
			} else {
				o1, o2 := c.Crossover(sub1, sub2)
				off1[k] = o1
				off2[k] = o2
			}
		}
		return off1, off2
	}

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
	if comp1, ok := p1.(CompositeGenome); ok {
		comp2 := p2.(CompositeGenome)
		off1 := make(CompositeGenome)
		off2 := make(CompositeGenome)
		for k, sub1 := range comp1 {
			sub2 := comp2[k]
			if _, isSeq := sub1.(SequenceGenome); isSeq {
				o1, o2 := PMXCrossover{}.Crossover(sub1, sub2)
				off1[k] = o1
				off2[k] = o2
			} else {
				o1, o2 := c.Crossover(sub1, sub2)
				off1[k] = o1
				off2[k] = o2
			}
		}
		return off1, off2
	}

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
	if comp1, ok := p1.(CompositeGenome); ok {
		comp2 := p2.(CompositeGenome)
		off1 := make(CompositeGenome)
		off2 := make(CompositeGenome)
		for k, sub1 := range comp1 {
			sub2 := comp2[k]
			if _, isSeq := sub1.(SequenceGenome); isSeq {
				o1, o2 := PMXCrossover{}.Crossover(sub1, sub2)
				off1[k] = o1
				off2[k] = o2
			} else {
				o1, o2 := c.Crossover(sub1, sub2)
				off1[k] = o1
				off2[k] = o2
			}
		}
		return off1, off2
	}

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

// PMXCrossover performs Partially Mapped Crossover (PMX) on two SequenceGenomes, preserving duplicate-free permutations.
type PMXCrossover struct{}

func (c PMXCrossover) Crossover(p1, p2 Genome) (Genome, Genome) {
	seq1, ok1 := p1.(SequenceGenome)
	seq2, ok2 := p2.(SequenceGenome)
	if !ok1 || !ok2 {
		return p1.Copy(), p2.Copy()
	}

	size := len(seq1)
	if size <= 1 {
		return p1.Copy(), p2.Copy()
	}

	point1 := rand.Intn(size - 1)
	point2 := rand.Intn(size-1-point1) + point1 + 1

	off1 := make(SequenceGenome, size)
	off2 := make(SequenceGenome, size)

	// Swapped segment copy
	copy(off1[point1:point2], seq2[point1:point2])
	copy(off2[point1:point2], seq1[point1:point2])

	// Crossover segment value mappings
	map1 := make(map[int]int)
	map2 := make(map[int]int)
	for i := point1; i < point2; i++ {
		map1[seq2[i]] = seq1[i]
		map2[seq1[i]] = seq2[i]
	}

	resolve1 := func(val int) int {
		for {
			mapped, exists := map1[val]
			if !exists {
				return val
			}
			val = mapped
		}
	}
	resolve2 := func(val int) int {
		for {
			mapped, exists := map2[val]
			if !exists {
				return val
			}
			val = mapped
		}
	}

	for i := 0; i < size; i++ {
		if i >= point1 && i < point2 {
			continue
		}
		off1[i] = resolve1(seq1[i])
		off2[i] = resolve2(seq2[i])
	}

	return off1, off2
}

// DefaultCrossoverer performs dynamic fallback crossover based on genome type.
type DefaultCrossoverer struct{}

func (c DefaultCrossoverer) Crossover(p1, p2 Genome) (Genome, Genome) {
	switch g1 := p1.(type) {
	case FloatGenome:
		return ArithmeticCrossover{Alpha: 0.5}.Crossover(p1, p2)
	case SequenceGenome:
		return PMXCrossover{}.Crossover(p1, p2)
	case CompositeGenome:
		g2 := p2.(CompositeGenome)
		off1 := make(CompositeGenome)
		off2 := make(CompositeGenome)
		for k, sub1 := range g1 {
			sub2 := g2[k]
			o1, o2 := c.Crossover(sub1, sub2)
			off1[k] = o1
			off2[k] = o2
		}
		return off1, off2
	default:
		return SinglePointCrossover{}.Crossover(p1, p2)
	}
}

// OrderCrossover performs Order Crossover (OX) on two SequenceGenomes, preserving duplicate-free permutations.
type OrderCrossover struct{}

// Crossover implements the Crossoverer interface, delegating composite genomes automatically.
func (c OrderCrossover) Crossover(p1, p2 Genome) (Genome, Genome) {
	if comp1, ok := p1.(CompositeGenome); ok {
		comp2 := p2.(CompositeGenome)
		off1 := make(CompositeGenome)
		off2 := make(CompositeGenome)
		for k, sub1 := range comp1 {
			sub2 := comp2[k]
			if _, isSeq := sub1.(SequenceGenome); isSeq {
				o1, o2 := c.Crossover(sub1, sub2)
				off1[k] = o1
				off2[k] = o2
			} else {
				o1, o2 := SinglePointCrossover{}.Crossover(sub1, sub2)
				off1[k] = o1
				off2[k] = o2
			}
		}
		return off1, off2
	}

	seq1, ok1 := p1.(SequenceGenome)
	seq2, ok2 := p2.(SequenceGenome)
	if !ok1 || !ok2 {
		return p1.Copy(), p2.Copy()
	}

	size := len(seq1)
	if size <= 1 {
		return p1.Copy(), p2.Copy()
	}

	point1 := rand.Intn(size - 1)
	point2 := rand.Intn(size-1-point1) + point1 + 1

	off1 := make(SequenceGenome, size)
	off2 := make(SequenceGenome, size)

	for i := 0; i < size; i++ {
		off1[i] = -1
		off2[i] = -1
	}

	copy(off1[point1:point2], seq1[point1:point2])
	copy(off2[point1:point2], seq2[point1:point2])

	fillRemaining := func(off, parent SequenceGenome, p1, p2 int) {
		size := len(parent)
		seen := make(map[int]bool)
		for i := p1; i < p2; i++ {
			seen[off[i]] = true
		}

		parentIdx := p2 % size
		offIdx := p2 % size

		for filledCount := 0; filledCount < size-(p2-p1); {
			val := parent[parentIdx]
			if !seen[val] {
				off[offIdx] = val
				offIdx = (offIdx + 1) % size
				filledCount++
			}
			parentIdx = (parentIdx + 1) % size
		}
	}

	fillRemaining(off1, seq2, point1, point2)
	fillRemaining(off2, seq1, point1, point2)

	return off1, off2
}

// CycleCrossover performs Cycle Crossover (CX) on two SequenceGenomes, preserving absolute position mappings.
type CycleCrossover struct{}

// Crossover implements the Crossoverer interface, delegating composite genomes automatically.
func (c CycleCrossover) Crossover(p1, p2 Genome) (Genome, Genome) {
	if comp1, ok := p1.(CompositeGenome); ok {
		comp2 := p2.(CompositeGenome)
		off1 := make(CompositeGenome)
		off2 := make(CompositeGenome)
		for k, sub1 := range comp1 {
			sub2 := comp2[k]
			if _, isSeq := sub1.(SequenceGenome); isSeq {
				o1, o2 := c.Crossover(sub1, sub2)
				off1[k] = o1
				off2[k] = o2
			} else {
				o1, o2 := SinglePointCrossover{}.Crossover(sub1, sub2)
				off1[k] = o1
				off2[k] = o2
			}
		}
		return off1, off2
	}

	seq1, ok1 := p1.(SequenceGenome)
	seq2, ok2 := p2.(SequenceGenome)
	if !ok1 || !ok2 {
		return p1.Copy(), p2.Copy()
	}

	size := len(seq1)
	if size <= 1 {
		return p1.Copy(), p2.Copy()
	}

	off1 := make(SequenceGenome, size)
	off2 := make(SequenceGenome, size)

	visited := make([]bool, size)

	cycleIndices := make(map[int]bool)
	currIdx := 0
	for {
		cycleIndices[currIdx] = true
		visited[currIdx] = true

		val2 := seq2[currIdx]
		nextIdx := -1
		for i, v := range seq1 {
			if v == val2 {
				nextIdx = i
				break
			}
		}

		if nextIdx == -1 || cycleIndices[nextIdx] {
			break
		}
		currIdx = nextIdx
	}

	for i := 0; i < size; i++ {
		if cycleIndices[i] {
			off1[i] = seq1[i]
			off2[i] = seq2[i]
		} else {
			off1[i] = seq2[i]
			off2[i] = seq1[i]
		}
	}

	return off1, off2
}

// EdgeRecombinationCrossover performs Edge Recombination Crossover (ERX) on two SequenceGenomes, preserving path linkages.
type EdgeRecombinationCrossover struct{}

// Crossover implements the Crossoverer interface, delegating composite genomes automatically.
func (c EdgeRecombinationCrossover) Crossover(p1, p2 Genome) (Genome, Genome) {
	if comp1, ok := p1.(CompositeGenome); ok {
		comp2 := p2.(CompositeGenome)
		off1 := make(CompositeGenome)
		off2 := make(CompositeGenome)
		for k, sub1 := range comp1 {
			sub2 := comp2[k]
			if _, isSeq := sub1.(SequenceGenome); isSeq {
				o1, o2 := c.Crossover(sub1, sub2)
				off1[k] = o1
				off2[k] = o2
			} else {
				o1, o2 := SinglePointCrossover{}.Crossover(sub1, sub2)
				off1[k] = o1
				off2[k] = o2
			}
		}
		return off1, off2
	}

	seq1, ok1 := p1.(SequenceGenome)
	seq2, ok2 := p2.(SequenceGenome)
	if !ok1 || !ok2 {
		return p1.Copy(), p2.Copy()
	}

	size := len(seq1)
	if size <= 1 {
		return p1.Copy(), p2.Copy()
	}

	off1 := c.erxSingle(seq1, seq2, seq1[0])
	off2 := c.erxSingle(seq1, seq2, seq2[0])

	return off1, off2
}

func (c EdgeRecombinationCrossover) erxSingle(seq1, seq2 SequenceGenome, startNode int) SequenceGenome {
	size := len(seq1)
	neighbors := make(map[int]map[int]bool)
	for _, v := range seq1 {
		neighbors[v] = make(map[int]bool)
	}

	addEdge := func(val, neighbor int) {
		neighbors[val][neighbor] = true
	}

	for i := 0; i < size; i++ {
		prev := seq1[(i-1+size)%size]
		next := seq1[(i+1)%size]
		addEdge(seq1[i], prev)
		addEdge(seq1[i], next)
	}

	for i := 0; i < size; i++ {
		prev := seq2[(i-1+size)%size]
		next := seq2[(i+1)%size]
		addEdge(seq2[i], prev)
		addEdge(seq2[i], next)
	}

	off := make(SequenceGenome, size)
	visited := make(map[int]bool)
	curr := startNode

	for step := 0; step < size; step++ {
		off[step] = curr
		visited[curr] = true

		for k := range neighbors {
			delete(neighbors[k], curr)
		}

		if step == size-1 {
			break
		}

		currNeighbors := neighbors[curr]
		next := -1

		if len(currNeighbors) > 0 {
			minEdges := 9999
			var candidates []int

			for nbr := range currNeighbors {
				edgesCount := len(neighbors[nbr])
				if edgesCount < minEdges {
					minEdges = edgesCount
					candidates = []int{nbr}
				} else if edgesCount == minEdges {
					candidates = append(candidates, nbr)
				}
			}

			if len(candidates) > 0 {
				next = candidates[rand.Intn(len(candidates))]
			}
		}

		if next == -1 {
			var unvisited []int
			for _, v := range seq1 {
				if !visited[v] {
					unvisited = append(unvisited, v)
				}
			}
			if len(unvisited) > 0 {
				next = unvisited[rand.Intn(len(unvisited))]
			}
		}

		curr = next
	}

	return off
}
