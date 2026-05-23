package nucleotide

import (
	"fmt"
	"math"
	"sort"
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
	if e.Config.AdaptiveMutation {
		diversity := e.genotypicDiversity()
		globalScaler = 1.0 + (1.0-diversity)*(e.Config.MaxMutationScaler-1.0)
	}

	// Fill the rest of the population
	for len(newPop) < e.Config.PopulationSize {
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

		totalScaler := globalScaler * ageScaler
		mut1Scaled := e.scaleMutator(mut1, totalScaler)
		mut2Scaled := e.scaleMutator(mut2, totalScaler)

		off1G = mut1Scaled.Mutate(off1G)
		off2G = mut2Scaled.Mutate(off2G)

		newPop = append(newPop, NewIndividual[Env, State](off1G))
		if len(newPop) < e.Config.PopulationSize {
			newPop = append(newPop, NewIndividual[Env, State](off2G))
		}
	}

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
	if e.Config.AdaptiveMutation {
		diversity := e.genotypicDiversity()
		globalScaler = 1.0 + (1.0-diversity)*(e.Config.MaxMutationScaler-1.0)
	}

	// 1. Generate offspring population Q_t of size N using selection, crossover, and mutation
	offspring := make(Population[Env, State], 0, len(current))
	for len(offspring) < len(current) {
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

		totalScaler := globalScaler * ageScaler
		mut1Scaled := e.scaleMutator(mut1, totalScaler)
		mut2Scaled := e.scaleMutator(mut2, totalScaler)

		off1G = mut1Scaled.Mutate(off1G)
		off2G = mut2Scaled.Mutate(off2G)

		off1 := NewIndividual[Env, State](off1G)
		off1.Fitness = e.Config.FitnessFunc(off1.Genome, e.Config.Env)
		offspring = append(offspring, off1)

		if len(offspring) < len(current) {
			off2 := NewIndividual[Env, State](off2G)
			off2.Fitness = e.Config.FitnessFunc(off2.Genome, e.Config.Env)
			offspring = append(offspring, off2)
		}
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
