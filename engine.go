package nucleotide

import (
	"fmt"
	"math/rand"
	"sort"
)

// FitnessFunc defines how to evaluate an individual's fitness across one or more objectives, with access to the environment.
type FitnessFunc[E any, S any] func(g Genome, env E) []float64

// ObjectiveDirection represents the optimization direction for a single objective.
type ObjectiveDirection int

const (
	Maximize ObjectiveDirection = iota
	Minimize
)

// ElitismFunc defines the strategy for carrying over individuals to the next generation.
type ElitismFunc[E any, S any] func(pop Population[E, S], size int) Population[E, S]

// BestIndividualElitism carries over the best individual.
func BestIndividualElitism[E any, S any](pop Population[E, S], size int) Population[E, S] {
	if size <= 0 || len(pop) == 0 {
		return nil
	}
	best := pop.Best()
	return Population[E, S]{NewIndividual[E, S](best.Genome.Copy())}
}

// TopNElitism sorts the population and carries over the top N individuals.
func TopNElitism[E any, S any](pop Population[E, S], size int) Population[E, S] {
	if size <= 0 || len(pop) == 0 {
		return nil
	}
	sortedPop := make(Population[E, S], len(pop))
	copy(sortedPop, pop)
	sort.Slice(sortedPop, func(i, j int) bool {
		var f1, f2 float64
		if len(sortedPop[i].Fitness) > 0 {
			f1 = sortedPop[i].Fitness[0]
		}
		if len(sortedPop[j].Fitness) > 0 {
			f2 = sortedPop[j].Fitness[0]
		}
		return f1 > f2
	})

	if size > len(sortedPop) {
		size = len(sortedPop)
	}

	result := make(Population[E, S], size)
	for i := 0; i < size; i++ {
		result[i] = NewIndividual[E, S](sortedPop[i].Genome.Copy())
	}
	return result
}

// EngineConfig holds the configuration for the evolution engine.
type EngineConfig[E any, S any] struct {
	PopulationSize     int
	MaxGenerations     int
	FitnessFunc        FitnessFunc[E, S]
	Selector           Selector
	Crossoverers       []Crossoverer
	Mutators           []Mutator
	CrossovererWeights []float64
	MutatorWeights     []float64
	Elitism            int
	ElitismFunc        ElitismFunc[E, S]
	PopulationFunc     PopulationFunc[E, S]
	Env                E
	ObjectiveDirections []ObjectiveDirection
	Strategy           GenerationStrategy[E, S]
}

// Engine orchestrates the genetic algorithm process.
type Engine[E any, S any] struct {
	Config       EngineConfig[E, S]
	Population   Population[E, S]
	Generation   int
	crossoverIdx int
	mutatorIdx   int
}

// NewEngine creates a new evolution engine and performs validation.
func NewEngine[E any, S any](config EngineConfig[E, S]) (*Engine[E, S], error) {
	if config.FitnessFunc == nil {
		return nil, fmt.Errorf("FitnessFunc must be defined in EngineConfig")
	}

	// Default Selector fallback
	if config.Selector == nil {
		config.Selector = GenericTournamentSelector[E, S]{Size: 3}
	}

	// Default Crossoverer fallback
	if len(config.Crossoverers) == 0 {
		config.Crossoverers = []Crossoverer{DefaultCrossoverer{}}
	}

	// Default Mutator fallback
	if len(config.Mutators) == 0 {
		config.Mutators = []Mutator{DefaultMutator{}}
	}

	// Validate CrossovererWeights
	if len(config.CrossovererWeights) > 0 {
		if len(config.CrossovererWeights) != len(config.Crossoverers) {
			return nil, fmt.Errorf("CrossovererWeights size (%d) must match Crossoverers size (%d)", len(config.CrossovererWeights), len(config.Crossoverers))
		}
		sum := 0.0
		for _, w := range config.CrossovererWeights {
			if w < 0 {
				return nil, fmt.Errorf("CrossovererWeights must not contain negative values")
			}
			sum += w
		}
		if sum == 0 {
			return nil, fmt.Errorf("sum of CrossovererWeights must be greater than zero")
		}
	}

	// Validate MutatorWeights
	if len(config.MutatorWeights) > 0 {
		if len(config.MutatorWeights) != len(config.Mutators) {
			return nil, fmt.Errorf("MutatorWeights size (%d) must match Mutators size (%d)", len(config.MutatorWeights), len(config.Mutators))
		}
		sum := 0.0
		for _, w := range config.MutatorWeights {
			if w < 0 {
				return nil, fmt.Errorf("MutatorWeights must not contain negative values")
			}
			sum += w
		}
		if sum == 0 {
			return nil, fmt.Errorf("sum of MutatorWeights must be greater than zero")
		}
	}

	if config.ElitismFunc == nil && config.Elitism > 0 {
		config.ElitismFunc = BestIndividualElitism[E, S]
	}
	if config.PopulationFunc == nil {
		config.PopulationFunc = DefaultPopulationFunc[E, S]
	}

	// Auto-deduce strategy if nil
	if config.Strategy == nil {
		if len(config.ObjectiveDirections) > 1 {
			config.Strategy = &NSGA2Generation[E, S]{}
		} else {
			config.Strategy = &StandardGeneration[E, S]{}
		}
	}

	e := &Engine[E, S]{
		Config: config,
	}

	if err := e.Config.Strategy.Initialize(e); err != nil {
		return nil, err
	}

	return e, nil
}

func (e *Engine[E, S]) selectCrossoverer() Crossoverer {
	n := len(e.Config.Crossoverers)
	if n == 0 {
		return DefaultCrossoverer{}
	}
	if n == 1 {
		return e.Config.Crossoverers[0]
	}

	if len(e.Config.CrossovererWeights) > 0 {
		totalWeight := 0.0
		for _, w := range e.Config.CrossovererWeights {
			totalWeight += w
		}
		r := rand.Float64() * totalWeight
		cumulative := 0.0
		for i, w := range e.Config.CrossovererWeights {
			cumulative += w
			if r <= cumulative {
				return e.Config.Crossoverers[i]
			}
		}
		return e.Config.Crossoverers[n-1]
	}

	// Round-robin selection
	idx := e.crossoverIdx
	e.crossoverIdx = (e.crossoverIdx + 1) % n
	return e.Config.Crossoverers[idx]
}

func (e *Engine[E, S]) selectMutator() Mutator {
	n := len(e.Config.Mutators)
	if n == 0 {
		return DefaultMutator{}
	}
	if n == 1 {
		return e.Config.Mutators[0]
	}

	if len(e.Config.MutatorWeights) > 0 {
		totalWeight := 0.0
		for _, w := range e.Config.MutatorWeights {
			totalWeight += w
		}
		r := rand.Float64() * totalWeight
		cumulative := 0.0
		for i, w := range e.Config.MutatorWeights {
			cumulative += w
			if r <= cumulative {
				return e.Config.Mutators[i]
			}
		}
		return e.Config.Mutators[n-1]
	}

	// Round-robin selection
	idx := e.mutatorIdx
	e.mutatorIdx = (e.mutatorIdx + 1) % n
	return e.Config.Mutators[idx]
}

// Run executes the genetic algorithm. It uses the provided definition to initialize the population if not already set.
func (e *Engine[E, S]) Run(def *Definition[E, S]) (*Individual[E, S], error) {
	if len(e.Population) == 0 {
		e.Population = e.Config.PopulationFunc(def, e.Config.PopulationSize)
	}
	e.Generation = 0

	// Initial evaluation
	e.evaluate()

	for e.Generation < e.Config.MaxGenerations {
		best := e.Population.Best()
		var bestFit []float64
		if best != nil {
			bestFit = best.Fitness
		}
		fmt.Printf("Generation %d: Best Fitness = %v, Avg Fitness = %v\n",
			e.Generation, bestFit, e.Population.AverageFitness())

		newPop, err := e.Config.Strategy.NextGeneration(e, def, e.Population)
		if err != nil {
			return nil, err
		}

		e.Population = newPop
		for _, ind := range e.Population {
			ind.Age++
		}
		e.Generation++
		e.evaluate()
	}

	return e.Population.Best(), nil
}

// ParetoFrontier returns all non-dominated individuals from the current population (Rank == 0).
func (e *Engine[E, S]) ParetoFrontier() Population[E, S] {
	if len(e.Population) == 0 {
		return nil
	}

	// Perform fast non-dominated sort to ensure ranks are computed/up to date
	fronts := fastNondominatedSort(e.Population, e.Config.ObjectiveDirections)
	if len(fronts) == 0 {
		return nil
	}

	frontier := make(Population[E, S], 0, len(fronts[0]))
	for _, idx := range fronts[0] {
		frontier = append(frontier, e.Population[idx])
	}
	return frontier
}

func (e *Engine[E, S]) evaluate() {
	for _, ind := range e.Population {
		ind.Fitness = e.Config.FitnessFunc(ind.Genome, e.Config.Env)
	}
}
