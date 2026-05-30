package nucleotide

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"sync"
)

// GenerationStrategy defines a modular execution interface for evolutionary generation loops.
type GenerationStrategy[Env any, State any] interface {
	Initialize(e *Engine[Env, State]) error
	NextGeneration(e *Engine[Env, State], def *Definition[Env, State], current Population[Env, State]) (Population[Env, State], error)
}

// StandardGeneration implements single-objective reproduction, crossover, mutation, and elitism replacement.
type StandardGeneration[Env any, State any] struct{}

func (s *StandardGeneration[Env, State]) Initialize(e *Engine[Env, State]) error {
	return nil
}

func (s *StandardGeneration[Env, State]) NextGeneration(e *Engine[Env, State], def *Definition[Env, State], current Population[Env, State]) (Population[Env, State], error) {
	newPop := make(Population[Env, State], 0, e.Config.PopulationSize)

	// Elitism
	if e.Config.ElitismFunc != nil && e.Config.Elitism > 0 {
		elites := e.Config.ElitismFunc(current, e.Config.Elitism)
		newPop = append(newPop, elites...)
	}

	globalScaler := 1.0
	if e.Config.AdaptiveMutation && e.Config.MutationController != nil {
		globalScaler = e.Config.MutationController.GetMutationScaler(e)
	}

	targetSize := e.Config.PopulationSize - len(newPop)
	if targetSize <= 0 {
		if len(newPop) > e.Config.PopulationSize {
			newPop = newPop[:e.Config.PopulationSize]
		}
		return newPop, nil
	}

	limit := e.effectiveConcurrencyLimit()
	if e.Config.DisableParallelReproduction || limit <= 1 || targetSize <= 2 {
		// Sequential pathway fallback
		for len(newPop) < e.Config.PopulationSize {
			off1, off2 := breed(e, current, globalScaler)
			newPop = append(newPop, off1)
			if len(newPop) < e.Config.PopulationSize {
				newPop = append(newPop, off2)
			}
		}
		return newPop, nil
	}

	// Concurrent chunked pathway
	workerCount := limit
	if workerCount > targetSize {
		workerCount = targetSize
	}

	chunks := make([]int, workerCount)
	baseChunk := targetSize / workerCount
	remainder := targetSize % workerCount
	for i := 0; i < workerCount; i++ {
		chunks[i] = baseChunk
		if remainder > 0 {
			chunks[i]++
			remainder--
		}
	}

	var wg sync.WaitGroup
	offspringMutex := sync.Mutex{}
	var allOffspring Population[Env, State]

	for w := 0; w < workerCount; w++ {
		wg.Add(1)
		chunkSize := chunks[w]
		go func() {
			defer wg.Done()
			localOffspring := make(Population[Env, State], 0, chunkSize)

			for len(localOffspring) < chunkSize {
				off1, off2 := breed(e, current, globalScaler)
				localOffspring = append(localOffspring, off1)
				if len(localOffspring) < chunkSize {
					localOffspring = append(localOffspring, off2)
				}
			}

			offspringMutex.Lock()
			allOffspring = append(allOffspring, localOffspring...)
			offspringMutex.Unlock()
		}()
	}
	wg.Wait()

	newPop = append(newPop, allOffspring...)
	if len(newPop) > e.Config.PopulationSize {
		newPop = newPop[:e.Config.PopulationSize]
	}

	return newPop, nil
}

// NSGA2Generation implements the multi-objective Non-dominated Sorting Genetic Algorithm II strategy.
type NSGA2Generation[Env any, State any] struct{}

func (s *NSGA2Generation[Env, State]) Initialize(e *Engine[Env, State]) error {
	return nil
}

func (s *NSGA2Generation[Env, State]) NextGeneration(e *Engine[Env, State], def *Definition[Env, State], current Population[Env, State]) (Population[Env, State], error) {
	if len(current) == 0 {
		return nil, fmt.Errorf("NSGA-II: parent population is empty")
	}

	globalScaler := 1.0
	if e.Config.AdaptiveMutation && e.Config.MutationController != nil {
		globalScaler = e.Config.MutationController.GetMutationScaler(e)
	}

	// 1. Generate offspring population Q_t of size N using selection, crossover, and mutation
	offspring := make(Population[Env, State], 0, len(current))
	targetSize := len(current)
	limit := e.effectiveConcurrencyLimit()

	if e.Config.DisableParallelReproduction || limit <= 1 || targetSize <= 2 {
		// Sequential pathway fallback
		for len(offspring) < len(current) {
			off1, off2 := breed(e, current, globalScaler)
			off1.Fitness = e.Config.FitnessFunc(off1.Genome, e.Config.Env)
			offspring = append(offspring, off1)

			if len(offspring) < len(current) {
				off2.Fitness = e.Config.FitnessFunc(off2.Genome, e.Config.Env)
				offspring = append(offspring, off2)
			}
		}
	} else {
		// Concurrent chunked pathway
		workerCount := limit
		if workerCount > targetSize {
			workerCount = targetSize
		}

		chunks := make([]int, workerCount)
		baseChunk := targetSize / workerCount
		remainder := targetSize % workerCount
		for i := 0; i < workerCount; i++ {
			chunks[i] = baseChunk
			if remainder > 0 {
				chunks[i]++
				remainder--
			}
		}

		var wg sync.WaitGroup
		offspringMutex := sync.Mutex{}

		for w := 0; w < workerCount; w++ {
			wg.Add(1)
			chunkSize := chunks[w]
			go func() {
				defer wg.Done()
				localOffspring := make(Population[Env, State], 0, chunkSize)

				for len(localOffspring) < chunkSize {
					off1, off2 := breed(e, current, globalScaler)
					off1.Fitness = e.Config.FitnessFunc(off1.Genome, e.Config.Env)
					localOffspring = append(localOffspring, off1)

					if len(localOffspring) < chunkSize {
						off2.Fitness = e.Config.FitnessFunc(off2.Genome, e.Config.Env)
						localOffspring = append(localOffspring, off2)
					}
				}

				offspringMutex.Lock()
				offspring = append(offspring, localOffspring...)
				offspringMutex.Unlock()
			}()
		}
		wg.Wait()
	}

	// 2. Combine parent population P_t and offspring Q_t to form R_t of size 2N
	combined := make(Population[Env, State], 0, len(current)+len(offspring))
	combined = append(combined, current...)
	combined = append(combined, offspring...)

	// 3. Fast non-dominated sorting of the combined population
	fronts := fastNondominatedSort(combined, e.Config.ObjectiveDirections)

	// 4. Fill the next generation population sequentially using the Pareto fronts
	newPop := make(Population[Env, State], 0, len(current))
	for _, frontIndices := range fronts {
		if len(frontIndices) == 0 {
			continue
		}

		// Calculate crowding distances for the current front
		calculateCrowdingDistances(combined, frontIndices, e.Config.ObjectiveDirections)

		if len(newPop)+len(frontIndices) <= len(current) {
			for _, idx := range frontIndices {
				newPop = append(newPop, combined[idx])
			}
		} else {
			// Sort the boundary front in descending order of crowding distance
			sort.Slice(frontIndices, func(i, j int) bool {
				return combined[frontIndices[i]].CrowdingDistance > combined[frontIndices[j]].CrowdingDistance
			})

			remaining := len(current) - len(newPop)
			for i := 0; i < remaining; i++ {
				newPop = append(newPop, combined[frontIndices[i]])
			}
			break
		}
	}

	return newPop, nil
}

// dominates returns true if ind1 dominates ind2.
// An individual ind1 dominates ind2 if it is no worse in all objectives
// and strictly better in at least one objective.
func dominates[Env any, State any](ind1, ind2 *Individual[Env, State], directions []ObjectiveDirection) bool {
	numObj := len(ind1.Fitness)
	if len(ind2.Fitness) < numObj {
		numObj = len(ind2.Fitness)
	}

	betterInAtLeastOne := false
	for i := 0; i < numObj; i++ {
		dir := Maximize
		if i < len(directions) {
			dir = directions[i]
		}

		f1 := ind1.Fitness[i]
		f2 := ind2.Fitness[i]

		if dir == Maximize {
			if f1 < f2 {
				return false
			}
			if f1 > f2 {
				betterInAtLeastOne = true
			}
		} else {
			if f1 > f2 {
				return false
			}
			if f1 < f2 {
				betterInAtLeastOne = true
			}
		}
	}
	return betterInAtLeastOne
}

// fastNondominatedSort groups individuals into dominance-based frontiers.
func fastNondominatedSort[Env any, State any](pop Population[Env, State], directions []ObjectiveDirection) [][]int {
	n := len(pop)
	if n == 0 {
		return nil
	}

	dominationSets := make([][]int, n)
	dominateCount := make([]int, n)

	var fronts [][]int
	var currentFront []int

	for i := 0; i < n; i++ {
		dominationSets[i] = make([]int, 0)
		dominateCount[i] = 0

		for j := 0; j < n; j++ {
			if i == j {
				continue
			}
			if dominates(pop[i], pop[j], directions) {
				dominationSets[i] = append(dominationSets[i], j)
			} else if dominates(pop[j], pop[i], directions) {
				dominateCount[i]++
			}
		}

		if dominateCount[i] == 0 {
			pop[i].Rank = 0
			currentFront = append(currentFront, i)
		}
	}

	fronts = append(fronts, currentFront)

	frontIdx := 0
	for len(fronts[frontIdx]) > 0 {
		var nextFront []int
		for _, pIdx := range fronts[frontIdx] {
			for _, qIdx := range dominationSets[pIdx] {
				dominateCount[qIdx]--
				if dominateCount[qIdx] == 0 {
					pop[qIdx].Rank = frontIdx + 1
					nextFront = append(nextFront, qIdx)
				}
			}
		}
		frontIdx++
		if len(nextFront) == 0 {
			break
		}
		fronts = append(fronts, nextFront)
	}

	return fronts
}

// calculateCrowdingDistances computes the sparsity metric for individuals in a front.
func calculateCrowdingDistances[Env any, State any](pop Population[Env, State], frontIndices []int, directions []ObjectiveDirection) {
	n := len(frontIndices)
	if n == 0 {
		return
	}
	if n <= 2 {
		for _, idx := range frontIndices {
			pop[idx].CrowdingDistance = math.MaxFloat64
		}
		return
	}

	// Initialize crowding distance to 0
	for _, idx := range frontIndices {
		pop[idx].CrowdingDistance = 0.0
	}

	numObjectives := len(pop[frontIndices[0]].Fitness)
	if numObjectives == 0 {
		return
	}

	for m := 0; m < numObjectives; m++ {
		// Sort frontIndices based on objective m
		sort.Slice(frontIndices, func(i, j int) bool {
			f1 := pop[frontIndices[i]].Fitness[m]
			f2 := pop[frontIndices[j]].Fitness[m]
			return f1 < f2
		})

		// Boundary solutions get infinite distance
		pop[frontIndices[0]].CrowdingDistance = math.MaxFloat64
		pop[frontIndices[n-1]].CrowdingDistance = math.MaxFloat64

		minFit := pop[frontIndices[0]].Fitness[m]
		maxFit := pop[frontIndices[n-1]].Fitness[m]
		diff := maxFit - minFit

		if diff > 0.0 {
			for i := 1; i < n-1; i++ {
				// Only add if the current distance is not already infinity
				if pop[frontIndices[i]].CrowdingDistance != math.MaxFloat64 {
					dist := (pop[frontIndices[i+1]].Fitness[m] - pop[frontIndices[i-1]].Fitness[m]) / diff
					pop[frontIndices[i]].CrowdingDistance += dist
				}
			}
		}
	}
}

// breed constructs offspring from two selected parents, applying age, adaptive, and self-adaptive mutation bounds.
func breed[Env any, State any](e *Engine[Env, State], current Population[Env, State], globalScaler float64) (*Individual[Env, State], *Individual[Env, State]) {
	p1 := e.Config.Selector.Select(current).(*Individual[Env, State])
	p2 := e.Config.Selector.Select(current).(*Individual[Env, State])

	cross := e.selectCrossoverer()
	mut1 := e.selectMutator()
	mut2 := e.selectMutator()

	off1G, off2G := cross.Crossover(p1.Genome, p2.Genome)

	ageScaler := 1.0
	if e.Config.AgeBiasedMutation {
		maxAge := p1.Age
		if p2.Age > maxAge {
			maxAge = p2.Age
		}
		if maxAge >= e.Config.AgeMutationThreshold {
			ageScaler = e.Config.AgeMutationScaler
		}
	}

	// Initialize individual mutation rates if zero
	if p1.MutationRate == 0 {
		p1.MutationRate = e.getBaseMutationRate(mut1)
	}
	if p2.MutationRate == 0 {
		p2.MutationRate = e.getBaseMutationRate(mut2)
	}

	childMutRate1 := 0.5 * (p1.MutationRate + p2.MutationRate)
	childMutRate2 := childMutRate1

	saCtrl, isSelfAdaptive := e.Config.MutationController.(*SelfAdaptiveController[Env, State])
	if e.Config.AdaptiveMutation && isSelfAdaptive && saCtrl != nil {
		factor1 := math.Exp(saCtrl.LearningRate * rand.NormFloat64())
		childMutRate1 *= factor1
		if childMutRate1 < saCtrl.MinRate {
			childMutRate1 = saCtrl.MinRate
		}
		if childMutRate1 > saCtrl.MaxRate {
			childMutRate1 = saCtrl.MaxRate
		}

		factor2 := math.Exp(saCtrl.LearningRate * rand.NormFloat64())
		childMutRate2 *= factor2
		if childMutRate2 < saCtrl.MinRate {
			childMutRate2 = saCtrl.MinRate
		}
		if childMutRate2 > saCtrl.MaxRate {
			childMutRate2 = saCtrl.MaxRate
		}
	}

	baseRate1 := e.getBaseMutationRate(mut1)
	var saScaler1 float64 = 1.0
	if isSelfAdaptive && baseRate1 > 0 {
		saScaler1 = childMutRate1 / baseRate1
	}

	totalScaler1 := globalScaler * ageScaler * saScaler1
	mut1Scaled := e.scaleMutator(mut1, totalScaler1)

	baseRate2 := e.getBaseMutationRate(mut2)
	var saScaler2 float64 = 1.0
	if isSelfAdaptive && baseRate2 > 0 {
		saScaler2 = childMutRate2 / baseRate2
	}

	totalScaler2 := globalScaler * ageScaler * saScaler2
	mut2Scaled := e.scaleMutator(mut2, totalScaler2)

	off1G = mut1Scaled.Mutate(off1G)
	off2G = mut2Scaled.Mutate(off2G)

	off1 := NewIndividual[Env, State](off1G)
	off1.MutationRate = childMutRate1

	off2 := NewIndividual[Env, State](off2G)
	off2.MutationRate = childMutRate2

	// Track parent fitnesses
	bestParentFitness := p1.Fitness
	if len(p2.Fitness) > 0 && len(bestParentFitness) > 0 {
		if len(e.Config.ObjectiveDirections) > 0 {
			offTemp2 := &Individual[Env, State]{Fitness: p2.Fitness}
			offTemp1 := &Individual[Env, State]{Fitness: p1.Fitness}
			if dominates(offTemp2, offTemp1, e.Config.ObjectiveDirections) {
				bestParentFitness = p2.Fitness
			}
		} else if p2.Fitness[0] > p1.Fitness[0] {
			bestParentFitness = p2.Fitness
		}
	}

	if len(bestParentFitness) > 0 {
		off1.ParentFitness = bestParentFitness
		off2.ParentFitness = bestParentFitness
	}

	return off1, off2
}
