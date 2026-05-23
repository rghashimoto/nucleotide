package nucleotide

import (
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"sort"
)

// FitnessFunc defines how to evaluate an individual's fitness across one or more objectives, with access to the environment.
type FitnessFunc[Env any, State any] func(g Genome, env Env) []float64

// WeightedCrossoverer pairs a crossoverer operator with its selection probability weight.
type WeightedCrossoverer struct {
	Crossoverer Crossoverer
	Weight      float64
}

// WeightedMutator pairs a mutator operator with its selection probability weight.
type WeightedMutator struct {
	Mutator Mutator
	Weight  float64
}

// ObjectiveDirection represents the optimization direction for a single objective.
type ObjectiveDirection int

const (
	Maximize ObjectiveDirection = iota
	Minimize
)

// ElitismFunc defines the strategy for carrying over individuals to the next generation.
type ElitismFunc[Env any, State any] func(pop Population[Env, State], size int) Population[Env, State]

// BestIndividualElitism carries over the best individual.
func BestIndividualElitism[Env any, State any](pop Population[Env, State], size int) Population[Env, State] {
	if size <= 0 || len(pop) == 0 {
		return nil
	}
	best := pop.Best()
	return Population[Env, State]{NewIndividual[Env, State](best.Genome.Copy())}
}

// TopNElitism sorts the population and carries over the top N individuals.
func TopNElitism[Env any, State any](pop Population[Env, State], size int) Population[Env, State] {
	if size <= 0 || len(pop) == 0 {
		return nil
	}
	sortedPop := make(Population[Env, State], len(pop))
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

	result := make(Population[Env, State], size)
	for i := 0; i < size; i++ {
		result[i] = NewIndividual[Env, State](sortedPop[i].Genome.Copy())
	}
	return result
}

// EngineConfig holds the configuration for the evolution engine.
type EngineConfig[Env any, State any] struct {
	PopulationSize       int
	MaxGenerations       int
	FitnessFunc          FitnessFunc[Env, State]
	Selector             Selector
	Crossoverers         []WeightedCrossoverer
	Mutators             []WeightedMutator
	Elitism              int
	ElitismFunc          ElitismFunc[Env, State]
	PopulationFunc       PopulationFunc[Env, State]
	Env                  Env
	ObjectiveDirections  []ObjectiveDirection
	Strategy             GenerationStrategy[Env, State]

	// Adaptive Mutation configuration
	AdaptiveMutation     bool
	MaxMutationScaler    float64
	OnMutationAdapted    func(generation int, diversity float64, currentScaler float64)

	// Age-Biased Mutation configuration
	AgeBiasedMutation    bool
	AgeMutationThreshold int
	AgeMutationScaler    float64
}

// Engine orchestrates the genetic algorithm process.
type Engine[Env any, State any] struct {
	Config             EngineConfig[Env, State]
	Population         Population[Env, State]
	Generation         int
	crossoverIdx       int
	mutatorIdx         int
	crossoverWeightSum float64
	mutatorWeightSum   float64
	DiversityHistory   []float64
}

// NewEngine creates a new evolution engine and performs validation.
func NewEngine[Env any, State any](config EngineConfig[Env, State]) (*Engine[Env, State], error) {
	if config.FitnessFunc == nil {
		return nil, fmt.Errorf("FitnessFunc must be defined in EngineConfig")
	}

	// Default Selector fallback
	if config.Selector == nil {
		config.Selector = GenericTournamentSelector[Env, State]{Size: 3}
	}

	// Default Crossoverer fallback
	if len(config.Crossoverers) == 0 {
		config.Crossoverers = []WeightedCrossoverer{{Crossoverer: DefaultCrossoverer{}}}
	}

	// Default Mutator fallback
	if len(config.Mutators) == 0 {
		config.Mutators = []WeightedMutator{{Mutator: DefaultMutator{}}}
	}

	// Validate Crossoverers and compute weight sum
	crossoverWeightSum := 0.0
	for _, wc := range config.Crossoverers {
		if wc.Weight < 0 {
			return nil, fmt.Errorf("crossoverer weight must not be negative")
		}
		crossoverWeightSum += wc.Weight
	}

	// Validate Mutators and compute weight sum
	mutatorWeightSum := 0.0
	for _, wm := range config.Mutators {
		if wm.Weight < 0 {
			return nil, fmt.Errorf("mutator weight must not be negative")
		}
		mutatorWeightSum += wm.Weight
	}

	if config.ElitismFunc == nil && config.Elitism > 0 {
		config.ElitismFunc = BestIndividualElitism[Env, State]
	}
	if config.PopulationFunc == nil {
		config.PopulationFunc = DefaultPopulationFunc[Env, State]
	}

	// Auto-deduce strategy if nil
	if config.Strategy == nil {
		if len(config.ObjectiveDirections) > 1 {
			config.Strategy = &NSGA2Generation[Env, State]{}
		} else {
			config.Strategy = &StandardGeneration[Env, State]{}
		}
	}

	e := &Engine[Env, State]{
		Config:             config,
		crossoverWeightSum: crossoverWeightSum,
		mutatorWeightSum:   mutatorWeightSum,
	}

	if e.Config.MaxMutationScaler <= 0 {
		e.Config.MaxMutationScaler = 3.0
	}
	if e.Config.AgeMutationThreshold <= 0 {
		e.Config.AgeMutationThreshold = 5
	}
	if e.Config.AgeMutationScaler <= 0 {
		e.Config.AgeMutationScaler = 2.0
	}

	if err := e.Config.Strategy.Initialize(e); err != nil {
		return nil, err
	}

	return e, nil
}

func (e *Engine[Env, State]) selectCrossoverer() Crossoverer {
	n := len(e.Config.Crossoverers)
	if n == 0 {
		return DefaultCrossoverer{}
	}
	if n == 1 {
		return e.Config.Crossoverers[0].Crossoverer
	}

	if e.crossoverWeightSum > 0 {
		r := rand.Float64() * e.crossoverWeightSum
		cumulative := 0.0
		for _, wc := range e.Config.Crossoverers {
			cumulative += wc.Weight
			if r <= cumulative {
				return wc.Crossoverer
			}
		}
		return e.Config.Crossoverers[n-1].Crossoverer
	}

	// Round-robin selection
	idx := e.crossoverIdx
	e.crossoverIdx = (e.crossoverIdx + 1) % n
	return e.Config.Crossoverers[idx].Crossoverer
}

func (e *Engine[Env, State]) selectMutator() Mutator {
	n := len(e.Config.Mutators)
	if n == 0 {
		return DefaultMutator{}
	}
	if n == 1 {
		return e.Config.Mutators[0].Mutator
	}

	if e.mutatorWeightSum > 0 {
		r := rand.Float64() * e.mutatorWeightSum
		cumulative := 0.0
		for _, wm := range e.Config.Mutators {
			cumulative += wm.Weight
			if r <= cumulative {
				return wm.Mutator
			}
		}
		return e.Config.Mutators[n-1].Mutator
	}

	// Round-robin selection
	idx := e.mutatorIdx
	e.mutatorIdx = (e.mutatorIdx + 1) % n
	return e.Config.Mutators[idx].Mutator
}

// Run executes the genetic algorithm. It uses the provided definition to initialize the population if not already set.
func (e *Engine[Env, State]) Run(def *Definition[Env, State]) (*Individual[Env, State], error) {
	if e.Config.PopulationSize == 0 {
		product := 1
		if def != nil && len(def.Loci) > 0 {
			for _, locus := range def.Loci {
				geneCount := len(locus.PossibleGenes)
				if geneCount > 0 {
					product *= geneCount
				}
			}
			e.Config.PopulationSize = 40 * product
		} else {
			e.Config.PopulationSize = 100 // Safe default fallback if no definition or loci defined
		}
	}

	if len(e.Population) == 0 {
		e.Population = e.Config.PopulationFunc(def, e.Config.PopulationSize)
	}
	e.Generation = 0

	// Initial evaluation
	e.evaluate()

	for e.Generation < e.Config.MaxGenerations {
		diversity := e.genotypicDiversity()
		e.DiversityHistory = append(e.DiversityHistory, diversity)

		if e.Config.AdaptiveMutation && e.Config.OnMutationAdapted != nil {
			scaler := 1.0 + (1.0-diversity)*(e.Config.MaxMutationScaler-1.0)
			e.Config.OnMutationAdapted(e.Generation, diversity, scaler)
		}

		best := e.Population.Best()
		var bestFit []float64
		if best != nil {
			bestFit = best.Fitness
		}
		fmt.Printf("Generation %d: Best Fitness = %v, Avg Fitness = %v, Diversity = %.1f%%\n",
			e.Generation, bestFit, e.Population.AverageFitness(), diversity*100.0)

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
func (e *Engine[Env, State]) ParetoFrontier() Population[Env, State] {
	if len(e.Population) == 0 {
		return nil
	}

	// Perform fast non-dominated sort to ensure ranks are computed/up to date
	fronts := fastNondominatedSort(e.Population, e.Config.ObjectiveDirections)
	if len(fronts) == 0 {
		return nil
	}

	frontier := make(Population[Env, State], 0, len(fronts[0]))
	for _, idx := range fronts[0] {
		frontier = append(frontier, e.Population[idx])
	}
	return frontier
}

func (e *Engine[Env, State]) evaluate() {
	for _, ind := range e.Population {
		ind.Fitness = e.Config.FitnessFunc(ind.Genome, e.Config.Env)
	}
}

func (e *Engine[Env, State]) genotypicDiversity() float64 {
	if len(e.Population) <= 1 {
		return 1.0
	}

	firstInd := e.Population[0]
	if firstInd == nil || firstInd.Genome == nil {
		return 1.0
	}

	switch g := firstInd.Genome.(type) {
	case *CategoricalGenome[Env, State]:
		size := g.Size()
		if size == 0 {
			return 1.0
		}
		totalDiv := 0.0
		for i := 0; i < size; i++ {
			counts := make(map[int]int)
			for _, ind := range e.Population {
				if cg, ok := ind.Genome.(*CategoricalGenome[Env, State]); ok {
					if i < len(cg.GeneIndices) {
						counts[cg.GeneIndices[i]]++
					}
				}
			}
			maxCount := 0
			for _, count := range counts {
				if count > maxCount {
					maxCount = count
				}
			}
			dominance := float64(maxCount) / float64(len(e.Population))
			totalDiv += (1.0 - dominance)
		}
		return totalDiv / float64(size)

	case BitGenome:
		size := g.Size()
		if size == 0 {
			return 1.0
		}
		totalDiv := 0.0
		for i := 0; i < size; i++ {
			trueCount := 0
			for _, ind := range e.Population {
				if bg, ok := ind.Genome.(BitGenome); ok {
					if i < len(bg) {
						if bg[i] {
							trueCount++
						}
					}
				}
			}
			falseCount := len(e.Population) - trueCount
			maxCount := trueCount
			if falseCount > maxCount {
				maxCount = falseCount
			}
			dominance := float64(maxCount) / float64(len(e.Population))
			totalDiv += (1.0 - dominance)
		}
		return totalDiv / float64(size)

	case FloatGenome:
		size := g.Size()
		if size == 0 {
			return 1.0
		}
		totalDiv := 0.0
		for i := 0; i < size; i++ {
			var validVals []float64
			for _, ind := range e.Population {
				if fg, ok := ind.Genome.(FloatGenome); ok {
					if i < len(fg) {
						validVals = append(validVals, fg[i])
					}
				}
			}
			if len(validVals) <= 1 {
				continue
			}

			minVal := validVals[0]
			maxVal := minVal
			sum := 0.0
			for _, v := range validVals {
				if v < minVal {
					minVal = v
				}
				if v > maxVal {
					maxVal = v
				}
				sum += v
			}

			if maxVal == minVal {
				continue
			}

			mean := sum / float64(len(validVals))
			varianceSum := 0.0
			for _, v := range validVals {
				diff := v - mean
				varianceSum += diff * diff
			}
			stdDev := math.Sqrt(varianceSum / float64(len(validVals)))

			// Scale standard deviation by maxVal - minVal
			locusDiv := (2.0 * stdDev) / (maxVal - minVal)
			if locusDiv > 1.0 {
				locusDiv = 1.0
			}
			totalDiv += locusDiv
		}
		return totalDiv / float64(size)

	default:
		return 1.0
	}
}

func (e *Engine[Env, State]) scaleMutator(m Mutator, scaler float64) Mutator {
	switch mut := m.(type) {
	case CategoricalMutator:
		mut.Probability *= scaler
		return mut
	case *CategoricalMutator:
		copied := *mut
		copied.Probability *= scaler
		return &copied
	case BitFlipMutator:
		mut.Probability *= scaler
		return mut
	case *BitFlipMutator:
		copied := *mut
		copied.Probability *= scaler
		return &copied
	case GaussianMutator:
		mut.Probability *= scaler
		return mut
	case *GaussianMutator:
		copied := *mut
		copied.Probability *= scaler
		return &copied
	case CategoricalCreepMutator:
		mut.Probability *= scaler
		return mut
	case *CategoricalCreepMutator:
		copied := *mut
		copied.Probability *= scaler
		return &copied
	case DefaultMutator:
		mut.Probability *= scaler
		return mut
	case *DefaultMutator:
		copied := *mut
		copied.Probability *= scaler
		return &copied
	default:
		return scaleMutatorReflection(m, scaler)
	}
}

func scaleMutatorReflection(m Mutator, scaler float64) Mutator {
	val := reflect.ValueOf(m)
	if val.Kind() == reflect.Ptr {
		valElem := val.Elem()
		if valElem.Kind() == reflect.Struct {
			copiedVal := reflect.New(valElem.Type())
			copiedVal.Elem().Set(valElem)

			probField := copiedVal.Elem().FieldByName("Probability")
			if probField.IsValid() && probField.CanSet() && probField.Kind() == reflect.Float64 {
				currentProb := probField.Float()
				probField.SetFloat(currentProb * scaler)
			}
			return copiedVal.Interface().(Mutator)
		}
	} else if val.Kind() == reflect.Struct {
		copiedPtr := reflect.New(val.Type())
		copiedPtr.Elem().Set(val)

		probField := copiedPtr.Elem().FieldByName("Probability")
		if probField.IsValid() && probField.CanSet() && probField.Kind() == reflect.Float64 {
			currentProb := probField.Float()
			probField.SetFloat(currentProb * scaler)
		}
		return copiedPtr.Elem().Interface().(Mutator)
	}
	return m
}
