package nucleotide

import (
	"math"
	"math/rand"
	"sort"
	"sync"
)

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

// GenericTournamentSelector is a type-safe selector for a specific environment Env and state State.
type GenericTournamentSelector[Env any, State any] struct {
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

	// Diversity-based Adaptive Sizing
	AdaptiveDiversity bool

	// Age-biased Selection
	AgeBias float64

	// Hall of Fame Competitor Integration
	HallOfFame            *Population[Env, State]
	HallOfFameProbability float64

	// Self-adaptive Selection
	SelfAdaptive bool
}

// NewProbabilisticTournamentSelector creates a tournament selector with selection probability controls.
func NewProbabilisticTournamentSelector[Env any, State any](size int, probability float64) GenericTournamentSelector[Env, State] {
	return GenericTournamentSelector[Env, State]{
		Size:        size,
		Probability: probability,
	}
}

// NewAdaptiveTournamentSelector creates a selector that dynamically scales tournament size.
func NewAdaptiveTournamentSelector[Env any, State any](minSize, maxSize int, progressFunc func() float64) GenericTournamentSelector[Env, State] {
	return GenericTournamentSelector[Env, State]{
		MinSize:            minSize,
		MaxSize:            maxSize,
		GenerationProgress: progressFunc,
	}
}

// NewNichingTournamentSelector creates a selector that applies local fitness sharing within tournaments.
func NewNichingTournamentSelector[Env any, State any](size int, sigma float64, distFunc func(g1, g2 Genome) float64) GenericTournamentSelector[Env, State] {
	return GenericTournamentSelector[Env, State]{
		Size:         size,
		SigmaShare:   sigma,
		DistanceFunc: distFunc,
	}
}

// NewUniqueTournamentSelector creates a selector that draws tournament competitors without replacement.
func NewUniqueTournamentSelector[Env any, State any](size int) GenericTournamentSelector[Env, State] {
	return GenericTournamentSelector[Env, State]{
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

func (s GenericTournamentSelector[Env, State]) Select(pop interface{}) interface{} {
	return s.SelectTyped(pop.(Population[Env, State]))
}

func (s GenericTournamentSelector[Env, State]) SelectTyped(pop Population[Env, State]) *Individual[Env, State] {
	n := len(pop)
	if n == 0 {
		return nil
	}

	// 1. Determine effective size (Adaptive Tournament & Diversity-based Sizing)
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
	} else if s.AdaptiveDiversity && n > 1 {
		var total float64
		for _, ind := range pop {
			var fit float64
			if len(ind.Fitness) > 0 {
				fit = ind.Fitness[0]
			}
			total += fit
		}
		avg := total / float64(n)

		var varianceSum float64
		for _, ind := range pop {
			var fit float64
			if len(ind.Fitness) > 0 {
				fit = ind.Fitness[0]
			}
			diff := fit - avg
			varianceSum += diff * diff
		}
		stdDev := math.Sqrt(varianceSum / float64(n))

		minSz := s.MinSize
		if minSz < 1 {
			minSz = 2
		}
		maxSz := s.MaxSize
		if maxSz < minSz {
			maxSz = s.Size
		}
		if maxSz < minSz {
			maxSz = minSz + 2
		}

		if avg > 0 {
			ratio := stdDev / avg
			if ratio > 1.0 {
				ratio = 1.0
			}
			size = minSz + int(ratio*float64(maxSz-minSz))
		} else {
			size = minSz
		}
	}

	if size < 1 {
		size = 1
	}
	if size > n {
		size = n
	}

	// 1.5 Self-adaptive Selection Size Override
	var tournament Population[Env, State]
	if s.SelfAdaptive && n > 1 {
		candidate := pop[rand.Intn(n)]
		k := s.Size

		type SelfAdaptiveIndividual interface {
			GetSelectionPreferences() (int, bool)
		}
		if sai, ok := interface{}(candidate).(SelfAdaptiveIndividual); ok {
			if preferredK, ok := sai.GetSelectionPreferences(); ok {
				k = preferredK
			}
		} else if sas, ok := interface{}(candidate.State).(SelfAdaptiveIndividual); ok {
			if preferredK, ok := sas.GetSelectionPreferences(); ok {
				k = preferredK
			}
		} else if cg, ok := candidate.Genome.(*CategoricalGenome[Env, State]); ok {
			for i, locus := range cg.Definition.Loci {
				if locus.ID == "TournamentSize" && locus.Type == LocusParameter {
					geneIdx := cg.GeneIndices[i]
					if val, ok := locus.PossibleGenes[geneIdx].Value.(int); ok {
						k = val
					}
				}
			}
		}

		if k < 1 {
			k = 1
		}
		if k > n {
			k = n
		}

		tournament = make(Population[Env, State], k)
		tournament[0] = candidate
		for i := 1; i < k; i++ {
			tournament[i] = pop[rand.Intn(n)]
		}
		size = k
	} else {
		// 2. Sample competitors (Unique Tournament & Hall of Fame Support)
		tournament = make(Population[Env, State], size)
		if s.Unique {
			selectedIndices := make(map[int]bool)
			for i := 0; i < size; i++ {
				if s.HallOfFame != nil && len(*s.HallOfFame) > 0 && rand.Float64() < s.HallOfFameProbability {
					hof := *s.HallOfFame
					tournament[i] = hof[rand.Intn(len(hof))]
					continue
				}

				if len(selectedIndices) >= n {
					tournament[i] = pop[rand.Intn(n)]
					continue
				}
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
				if s.HallOfFame != nil && len(*s.HallOfFame) > 0 && rand.Float64() < s.HallOfFameProbability {
					hof := *s.HallOfFame
					tournament[i] = hof[rand.Intn(len(hof))]
				} else {
					tournament[i] = pop[rand.Intn(n)]
				}
			}
		}
	}

	// 3. Fitness Sharing / Niching / Age Bias
	isMultiObjective := false
	if n > 0 && len(pop[0].Fitness) > 1 {
		isMultiObjective = true
	}

	type ratedCompetitor struct {
		ind         *Individual[Env, State]
		originalFit float64
		adjustedFit float64
	}
	competitors := make([]ratedCompetitor, size)
	for i, ind := range tournament {
		var fit float64
		if len(ind.Fitness) > 0 {
			fit = ind.Fitness[0]
		}
		competitors[i] = ratedCompetitor{
			ind:         ind,
			originalFit: fit,
			adjustedFit: fit,
		}
	}

	if isMultiObjective {
		// Sort competitors using the Crowded Comparison Operator (<_c)
		sort.Slice(competitors, func(i, j int) bool {
			indI := competitors[i].ind
			indJ := competitors[j].ind
			if indI.Rank != indJ.Rank {
				return indI.Rank < indJ.Rank
			}
			return indI.CrowdingDistance > indJ.CrowdingDistance
		})
	} else {
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

		if s.AgeBias != 0.0 {
			for i := 0; i < size; i++ {
				competitors[i].adjustedFit -= float64(competitors[i].ind.Age) * s.AgeBias
			}
		}

		// Sort by adjusted fitness descending
		sort.Slice(competitors, func(i, j int) bool {
			return competitors[i].adjustedFit > competitors[j].adjustedFit
		})
	}

	// 4. Selection (Probabilistic Selection Support)
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

// RouletteWheelSelector selects individuals proportional to their fitness.
type RouletteWheelSelector[Env any, State any] struct {
	AutoShift bool // If true, shifts negative fitnesses so minimum is 0.
}

func (s RouletteWheelSelector[Env, State]) Select(pop interface{}) interface{} {
	return s.SelectTyped(pop.(Population[Env, State]))
}

func (s RouletteWheelSelector[Env, State]) SelectTyped(pop Population[Env, State]) *Individual[Env, State] {
	n := len(pop)
	if n == 0 {
		return nil
	}
	if n == 1 {
		return pop[0]
	}

	fitnesses := make([]float64, n)
	var minFit float64
	if len(pop[0].Fitness) > 0 {
		minFit = pop[0].Fitness[0]
	}
	for i, ind := range pop {
		var fit float64
		if len(ind.Fitness) > 0 {
			fit = ind.Fitness[0]
		}
		fitnesses[i] = fit
		if fit < minFit {
			minFit = fit
		}
	}

	if s.AutoShift || minFit < 0 {
		shift := 0.0
		if minFit < 0 {
			shift = -minFit
		} else if s.AutoShift {
			shift = minFit
		}
		shift += 0.0001
		for i := range fitnesses {
			fitnesses[i] += shift
		}
	}

	sum := 0.0
	for _, f := range fitnesses {
		if f < 0 {
			f = 0
		}
		sum += f
	}

	if sum <= 0 {
		return pop[rand.Intn(n)]
	}

	r := rand.Float64() * sum
	cumulative := 0.0
	for i, f := range fitnesses {
		if f < 0 {
			f = 0
		}
		cumulative += f
		if r <= cumulative {
			return pop[i]
		}
	}

	return pop[n-1]
}

// StochasticUniversalSamplingSelector uses a single spin to select multiple parents with minimal bias.
type StochasticUniversalSamplingSelector[Env any, State any] struct {
	AutoShift bool
	queue     *[]*Individual[Env, State]
	mu        *sync.Mutex
}

func (s *StochasticUniversalSamplingSelector[Env, State]) Select(pop interface{}) interface{} {
	return s.SelectTyped(pop.(Population[Env, State]))
}

func (s *StochasticUniversalSamplingSelector[Env, State]) SelectTyped(pop Population[Env, State]) *Individual[Env, State] {
	n := len(pop)
	if n == 0 {
		return nil
	}

	if s.mu == nil {
		s.mu = &sync.Mutex{}
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.queue == nil {
		s.queue = &[]*Individual[Env, State]{}
	}

	if len(*s.queue) == 0 {
		s.fillQueue(pop)
	}

	if len(*s.queue) > 0 {
		val := (*s.queue)[0]
		*s.queue = (*s.queue)[1:]
		return val
	}

	return pop[rand.Intn(n)]
}

func (s *StochasticUniversalSamplingSelector[Env, State]) fillQueue(pop Population[Env, State]) {
	n := len(pop)
	if n == 0 {
		return
	}

	fitnesses := make([]float64, n)
	var minFit float64
	if len(pop[0].Fitness) > 0 {
		minFit = pop[0].Fitness[0]
	}
	for i, ind := range pop {
		var fit float64
		if len(ind.Fitness) > 0 {
			fit = ind.Fitness[0]
		}
		fitnesses[i] = fit
		if fit < minFit {
			minFit = fit
		}
	}

	if s.AutoShift || minFit < 0 {
		shift := 0.0
		if minFit < 0 {
			shift = -minFit
		} else if s.AutoShift {
			shift = minFit
		}
		shift += 0.0001
		for i := range fitnesses {
			fitnesses[i] += shift
		}
	}

	sum := 0.0
	for _, f := range fitnesses {
		if f < 0 {
			f = 0
		}
		sum += f
	}

	if sum <= 0 {
		q := make([]*Individual[Env, State], n)
		for i := 0; i < n; i++ {
			q[i] = pop[rand.Intn(n)]
		}
		*s.queue = q
		return
	}

	ptrDistance := sum / float64(n)
	start := rand.Float64() * ptrDistance

	q := make([]*Individual[Env, State], 0, n)
	currSum := 0.0
	idx := 0

	for i := 0; i < n; i++ {
		pointer := start + float64(i)*ptrDistance
		for currSum < pointer && idx < n {
			f := fitnesses[idx]
			if f < 0 {
				f = 0
			}
			currSum += f
			if currSum >= pointer {
				break
			}
			idx++
			if idx >= n {
				idx = n - 1
				break
			}
		}
		q = append(q, pop[idx])
	}

	*s.queue = q
}

// RankSelector selects individuals based on their fitness rank rather than absolute fitness.
type RankSelector[Env any, State any] struct {
	SelectionPressure float64 // typically in [1.0, 2.0], defaults to 1.5 if <= 0
}

func (s RankSelector[Env, State]) Select(pop interface{}) interface{} {
	return s.SelectTyped(pop.(Population[Env, State]))
}

func (s RankSelector[Env, State]) SelectTyped(pop Population[Env, State]) *Individual[Env, State] {
	n := len(pop)
	if n == 0 {
		return nil
	}
	if n == 1 {
		return pop[0]
	}

	sp := s.SelectionPressure
	if sp <= 0 {
		sp = 1.5
	}
	if sp < 1.0 {
		sp = 1.0
	}
	if sp > 2.0 {
		sp = 2.0
	}

	sortedPop := make(Population[Env, State], n)
	copy(sortedPop, pop)
	sort.Slice(sortedPop, func(i, j int) bool {
		var f1, f2 float64
		if len(sortedPop[i].Fitness) > 0 {
			f1 = sortedPop[i].Fitness[0]
		}
		if len(sortedPop[j].Fitness) > 0 {
			f2 = sortedPop[j].Fitness[0]
		}
		return f1 < f2
	})

	probs := make([]float64, n)
	sum := 0.0
	for i := 0; i < n; i++ {
		p := (2.0 - sp) / float64(n) + (2.0 * float64(i) * (sp - 1.0)) / float64(n * (n - 1))
		probs[i] = p
		sum += p
	}

	r := rand.Float64() * sum
	cumulative := 0.0
	for i, p := range probs {
		cumulative += p
		if r <= cumulative {
			return sortedPop[i]
		}
	}

	return sortedPop[n-1]
}

// BoltzmannSelector selects individuals using a Boltzmann distribution with temperature.
type BoltzmannSelector[Env any, State any] struct {
	Temperature float64 // Defaults to 1.0 if <= 0
}

func (s BoltzmannSelector[Env, State]) Select(pop interface{}) interface{} {
	return s.SelectTyped(pop.(Population[Env, State]))
}

func (s BoltzmannSelector[Env, State]) SelectTyped(pop Population[Env, State]) *Individual[Env, State] {
	n := len(pop)
	if n == 0 {
		return nil
	}
	if n == 1 {
		return pop[0]
	}

	temp := s.Temperature
	if temp <= 0 {
		temp = 1.0
	}

	var maxFit float64
	if len(pop[0].Fitness) > 0 {
		maxFit = pop[0].Fitness[0]
	}
	for _, ind := range pop {
		var fit float64
		if len(ind.Fitness) > 0 {
			fit = ind.Fitness[0]
		}
		if fit > maxFit {
			maxFit = fit
		}
	}

	probs := make([]float64, n)
	sum := 0.0
	for i, ind := range pop {
		var fit float64
		if len(ind.Fitness) > 0 {
			fit = ind.Fitness[0]
		}
		val := math.Exp((fit - maxFit) / temp)
		probs[i] = val
		sum += val
	}

	if sum <= 0 {
		return pop[rand.Intn(n)]
	}

	r := rand.Float64() * sum
	cumulative := 0.0
	for i, p := range probs {
		cumulative += p
		if r <= cumulative {
			return pop[i]
		}
	}

	return pop[n-1]
}
