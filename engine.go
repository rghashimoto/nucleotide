package nucleotide

import (
	"fmt"
	"sort"
)

// FitnessFunc defines how to evaluate an individual's fitness, with access to the environment.
type FitnessFunc[E any] func(g Genome, env E) float64

// ElitismFunc defines the strategy for carrying over individuals to the next generation.
type ElitismFunc[E any] func(pop Population[E], size int) Population[E]

// BestIndividualElitism carries over the best individual.
func BestIndividualElitism[E any](pop Population[E], size int) Population[E] {
	if size <= 0 || len(pop) == 0 {
		return nil
	}
	best := pop.Best()
	return Population[E]{NewIndividual[E](best.Genome.Copy())}
}

// TopNElitism sorts the population and carries over the top N individuals.
func TopNElitism[E any](pop Population[E], size int) Population[E] {
	if size <= 0 || len(pop) == 0 {
		return nil
	}
	sortedPop := make(Population[E], len(pop))
	copy(sortedPop, pop)
	sort.Slice(sortedPop, func(i, j int) bool {
		return sortedPop[i].Fitness > sortedPop[j].Fitness
	})

	if size > len(sortedPop) {
		size = len(sortedPop)
	}

	result := make(Population[E], size)
	for i := 0; i < size; i++ {
		result[i] = NewIndividual[E](sortedPop[i].Genome.Copy())
	}
	return result
}

// EngineConfig holds the configuration for the evolution engine.
type EngineConfig[E any] struct {
	PopulationSize int
	MaxGenerations int
	FitnessFunc    FitnessFunc[E]
	Selector       Selector
	Crossoverer    Crossoverer
	Mutator        Mutator
	Elitism        int
	ElitismFunc    ElitismFunc[E]
	PopulationFunc PopulationFunc[E] // User can provide their own
	Env            E
}

// Engine orchestrates the genetic algorithm process.
type Engine[E any] struct {
	Config     EngineConfig[E]
	Population Population[E]
	Generation int
}

// NewEngine creates a new evolution engine.
func NewEngine[E any](config EngineConfig[E]) *Engine[E] {
	if config.ElitismFunc == nil && config.Elitism > 0 {
		config.ElitismFunc = BestIndividualElitism[E]
	}
	if config.PopulationFunc == nil {
		config.PopulationFunc = DefaultPopulationFunc[E]
	}
	return &Engine[E]{
		Config: config,
	}
}

// Run executes the genetic algorithm. It uses the provided definition to initialize the population if not already set.
func (e *Engine[E]) Run(def *Definition[E]) (*Individual[E], error) {
	if len(e.Population) == 0 {
		e.Population = e.Config.PopulationFunc(def, e.Config.PopulationSize)
	}
	e.Generation = 0

	// Initial evaluation
	e.evaluate()

	for e.Generation < e.Config.MaxGenerations {
		fmt.Printf("Generation %d: Best Fitness = %.4f, Avg Fitness = %.4f\n",
			e.Generation, e.Population.Best().Fitness, e.Population.AverageFitness())

		newPop := make(Population[E], 0, e.Config.PopulationSize)

		// Elitism
		if e.Config.ElitismFunc != nil && e.Config.Elitism > 0 {
			elites := e.Config.ElitismFunc(e.Population, e.Config.Elitism)
			newPop = append(newPop, elites...)
		}

		// Fill the rest of the population
		for len(newPop) < e.Config.PopulationSize {
			p1 := e.Config.Selector.Select(e.Population).(*Individual[E])
			p2 := e.Config.Selector.Select(e.Population).(*Individual[E])

			off1G, off2G := e.Config.Crossoverer.Crossover(p1.Genome, p2.Genome)

			off1G = e.Config.Mutator.Mutate(off1G)
			off2G = e.Config.Mutator.Mutate(off2G)

			newPop = append(newPop, NewIndividual[E](off1G))
			if len(newPop) < e.Config.PopulationSize {
				newPop = append(newPop, NewIndividual[E](off2G))
			}
		}

		if len(newPop) > e.Config.PopulationSize {
			newPop = newPop[:e.Config.PopulationSize]
		}

		e.Population = newPop
		e.Generation++
		e.evaluate()
	}

	return e.Population.Best(), nil
}

func (e *Engine[E]) evaluate() {
	for _, ind := range e.Population {
		ind.Fitness = e.Config.FitnessFunc(ind.Genome, e.Config.Env)
	}
}
