package nucleotide

import (
	"math"
	"math/rand"
	"sort"
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

	// Probabilistic Selection
	Probability float64 // If > 0 and < 1.0, active. Best competitor has chance P, next has P*(1-P), etc.

	// Adaptive Selection
	MinSize            int
	MaxSize            int
	GenerationProgress func() float64 // Function returning fraction [0.0, 1.0] representing run progress.

	// Niching / Local Fitness Sharing
	SigmaShare   float64                     // Niching niche radius (active if > 0)
	NichingAlpha float64                     // Sharing power factor (defaults to 1.0 if <= 0)
	DistanceFunc func(g1, g2 Genome) float64 // Optional custom distance metric.

	// Unique Tournament
	Unique bool // If true, selects competitors without replacement.
}

// NewProbabilisticTournamentSelector creates a tournament selector with selection probability controls.
func NewProbabilisticTournamentSelector[E any, S any](size int, probability float64) GenericTournamentSelector[E, S] {
	return GenericTournamentSelector[E, S]{
		Size:        size,
		Probability: probability,
	}
}

// NewAdaptiveTournamentSelector creates a selector that dynamically scales tournament size.
func NewAdaptiveTournamentSelector[E any, S any](minSize, maxSize int, progressFunc func() float64) GenericTournamentSelector[E, S] {
	return GenericTournamentSelector[E, S]{
		MinSize:            minSize,
		MaxSize:            maxSize,
		GenerationProgress: progressFunc,
	}
}

// NewNichingTournamentSelector creates a selector that applies local fitness sharing within tournaments.
func NewNichingTournamentSelector[E any, S any](size int, sigma float64, distFunc func(g1, g2 Genome) float64) GenericTournamentSelector[E, S] {
	return GenericTournamentSelector[E, S]{
		Size:         size,
		SigmaShare:   sigma,
		DistanceFunc: distFunc,
	}
}

// NewUniqueTournamentSelector creates a selector that draws tournament competitors without replacement.
func NewUniqueTournamentSelector[E any, S any](size int) GenericTournamentSelector[E, S] {
	return GenericTournamentSelector[E, S]{
		Size:   size,
		Unique: true,
	}
}

func defaultGenomeDistance(g1, g2 Genome) float64 {
	switch genome1 := g1.(type) {
	case BitGenome:
		if genome2, ok := g2.(BitGenome); ok {
			diff := 0
			sz := genome1.Size()
			if sz == 0 {
				return 0
			}
			for i := 0; i < sz; i++ {
				if genome1[i] != genome2[i] {
					diff++
				}
			}
			return float64(diff) / float64(sz)
		}
	case FloatGenome:
		if genome2, ok := g2.(FloatGenome); ok {
			sz := genome1.Size()
			if sz == 0 {
				return 0
			}
			sum := 0.0
			for i := 0; i < sz; i++ {
				diff := genome1[i] - genome2[i]
				sum += diff * diff
			}
			return math.Sqrt(sum / float64(sz))
		}
	default:
		type indexedGenome interface {
			GetIndices() []int
		}
		if ig1, ok := g1.(indexedGenome); ok {
			if ig2, ok := g2.(indexedGenome); ok {
				indices1 := ig1.GetIndices()
				indices2 := ig2.GetIndices()
				sz := len(indices1)
				if sz == 0 {
					return 0
				}
				diff := 0
				for i := 0; i < sz; i++ {
					if indices1[i] != indices2[i] {
						diff++
					}
				}
				return float64(diff) / float64(sz)
			}
		}
	}
	return 0.0
}

func (s GenericTournamentSelector[E, S]) Select(pop interface{}) interface{} {
	return s.SelectTyped(pop.(Population[E, S]))
}

func (s GenericTournamentSelector[E, S]) SelectTyped(pop Population[E, S]) *Individual[E, S] {
	n := len(pop)
	if n == 0 {
		return nil
	}

	// 1. Determine effective size (Adaptive Tournament Support)
	size := s.Size
	if s.GenerationProgress != nil && s.MinSize > 0 && s.MaxSize > 0 {
		progress := s.GenerationProgress()
		if progress < 0 {
			progress = 0
		}
		if progress > 1 {
			progress = 1
		}
		size = int(float64(s.MinSize) + progress*float64(s.MaxSize-s.MinSize))
	}
	if size < 1 {
		size = 1
	}
	if size > n {
		size = n
	}

	// 2. Sample competitors (Unique Tournament Support)
	tournament := make(Population[E, S], size)
	if s.Unique {
		selectedIndices := make(map[int]bool)
		for i := 0; i < size; i++ {
			for {
				idx := rand.Intn(n)
				if !selectedIndices[idx] {
					selectedIndices[idx] = true
					tournament[i] = pop[idx]
					break
				}
			}
		}
	} else {
		for i := 0; i < size; i++ {
			tournament[i] = pop[rand.Intn(n)]
		}
	}

	// 3. Fitness Sharing / Niching
	type ratedCompetitor struct {
		ind         *Individual[E, S]
		originalFit float64
		adjustedFit float64
	}
	competitors := make([]ratedCompetitor, size)
	for i, ind := range tournament {
		competitors[i] = ratedCompetitor{
			ind:         ind,
			originalFit: ind.Fitness,
			adjustedFit: ind.Fitness,
		}
	}

	if s.SigmaShare > 0 {
		alpha := s.NichingAlpha
		if alpha <= 0 {
			alpha = 1.0
		}
		distFunc := s.DistanceFunc
		if distFunc == nil {
			distFunc = defaultGenomeDistance
		}

		for i := 0; i < size; i++ {
			nicheCount := 0.0
			for j := 0; j < size; j++ {
				dist := distFunc(competitors[i].ind.Genome, competitors[j].ind.Genome)
				if dist < s.SigmaShare {
					nicheCount += 1.0 - math.Pow(dist/s.SigmaShare, alpha)
				}
			}
			if nicheCount > 0 {
				competitors[i].adjustedFit = competitors[i].originalFit / nicheCount
			}
		}
	}

	// 4. Selection (Probabilistic Selection Support)
	sort.Slice(competitors, func(i, j int) bool {
		return competitors[i].adjustedFit > competitors[j].adjustedFit
	})

	if s.Probability > 0 && s.Probability < 1.0 {
		for i := 0; i < size; i++ {
			if rand.Float64() < s.Probability {
				return competitors[i].ind
			}
		}
		return competitors[size-1].ind
	}

	return competitors[0].ind
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
