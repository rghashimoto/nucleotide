package nucleotide

import (
	"fmt"
	"math/rand"
	"sort"
	"sync"
)

// MigrationTopology defines how migrants are routed between islands.
type MigrationTopology int

const (
	TopologyRing MigrationTopology = iota
	TopologyRandom
)

// MigrationPolicy defines which individuals migrate and who they replace.
type MigrationPolicy int

const (
	PolicyBestReplaceWorst MigrationPolicy = iota
	PolicyRandomReplaceRandom
)

// MultiIslandEngineConfig holds settings for MultiIslandEngine.
type MultiIslandEngineConfig[Env any, State any] struct {
	NumIslands        int
	MigrationInterval int
	MigrationRate     int
	MigrationTopology MigrationTopology
	MigrationPolicy   MigrationPolicy

	EngineConfig EngineConfig[Env, State]
	EnvFactory   func(islandIndex int) Env
}

// MultiIslandEngine orchestrates a parallel island model genetic algorithm.
type MultiIslandEngine[Env any, State any] struct {
	Config  MultiIslandEngineConfig[Env, State]
	Islands []*Engine[Env, State]
}

// NewMultiIslandEngine instantiates a MultiIslandEngine.
func NewMultiIslandEngine[Env any, State any](config MultiIslandEngineConfig[Env, State]) (*MultiIslandEngine[Env, State], error) {
	if config.NumIslands <= 0 {
		return nil, fmt.Errorf("number of islands must be greater than 0")
	}

	islands := make([]*Engine[Env, State], config.NumIslands)
	for i := 0; i < config.NumIslands; i++ {
		// Clone EngineConfig template
		islandConfig := config.EngineConfig

		// Determine Environment based on Options (A, B, or C)
		if config.EnvFactory != nil {
			// Option B & C: Build distinct Env instance for this island
			islandConfig.Env = config.EnvFactory(i)
		}
		// If EnvFactory is nil, falls back to Option A (using templates shared Env)

		engine, err := NewEngine(islandConfig)
		if err != nil {
			return nil, fmt.Errorf("failed to create island engine %d: %w", i, err)
		}
		islands[i] = engine
	}

	return &MultiIslandEngine[Env, State]{
		Config:  config,
		Islands: islands,
	}, nil
}

// Run executes the parallel island evolution.
func (m *MultiIslandEngine[Env, State]) Run(def *Definition[Env, State]) (*Individual[Env, State], error) {
	// Initialize all island engines
	for _, island := range m.Islands {
		if island.Config.PopulationSize == 0 {
			product := 1
			if def != nil && len(def.Loci) > 0 {
				for _, locus := range def.Loci {
					geneCount := len(locus.PossibleGenes)
					if geneCount > 0 {
						product *= geneCount
					}
				}
				island.Config.PopulationSize = 40 * product
			} else {
				island.Config.PopulationSize = 100
			}
		}

		if len(island.Population) == 0 {
			island.Population = island.Config.PopulationFunc(def, island.Config.PopulationSize)
			island.evaluate()
		}
		island.Generation = 0
	}

	maxGens := m.Config.EngineConfig.MaxGenerations
	interval := m.Config.MigrationInterval

	for currentGen := 0; currentGen < maxGens; {
		steps := interval
		if steps <= 0 {
			steps = maxGens // If no migration interval, run straight to max generations
		}
		if currentGen+steps > maxGens {
			steps = maxGens - currentGen
		}

		// 1. Run all islands concurrently for 'steps' generations
		var wg sync.WaitGroup
		errs := make([]error, m.Config.NumIslands)
		for i, island := range m.Islands {
			wg.Add(1)
			go func(idx int, eng *Engine[Env, State]) {
				defer wg.Done()
				for s := 0; s < steps; s++ {
					if err := eng.Step(def); err != nil {
						errs[idx] = err
						return
					}
				}
			}(i, island)
		}
		wg.Wait()

		for _, err := range errs {
			if err != nil {
				return nil, err
			}
		}

		currentGen += steps

		// 2. Perform migration if not finished and migration is enabled
		if currentGen < maxGens && interval > 0 {
			m.migrate()
		}
	}

	// Find the global best individual across all islands
	var globalBest *Individual[Env, State]
	for _, island := range m.Islands {
		best := island.Population.Best()
		if best != nil {
			if globalBest == nil || len(best.Fitness) > 0 && (len(globalBest.Fitness) == 0 || best.Fitness[0] > globalBest.Fitness[0]) {
				globalBest = best
			}
		}
	}

	return globalBest, nil
}

// migrate exchanges individuals between islands.
func (m *MultiIslandEngine[Env, State]) migrate() {
	numIslands := len(m.Islands)
	if numIslands <= 1 {
		return
	}

	rate := m.Config.MigrationRate
	if rate <= 0 {
		rate = 1 // Default to 1 migrant
	}

	// 1. Collect migrants from each island based on selection policy
	migrants := make([][]*Individual[Env, State], numIslands)

	for i, island := range m.Islands {
		popSize := len(island.Population)
		if popSize == 0 {
			continue
		}
		effRate := rate
		if effRate > popSize {
			effRate = popSize
		}

		selected := make([]*Individual[Env, State], effRate)
		if m.Config.MigrationPolicy == PolicyBestReplaceWorst {
			// Sort population (clone so we don't scramble original array indices during selection)
			cloned := make(Population[Env, State], popSize)
			copy(cloned, island.Population)
			sort.Slice(cloned, func(a, b int) bool {
				fitA := 0.0
				fitB := 0.0
				if len(cloned[a].Fitness) > 0 {
					fitA = cloned[a].Fitness[0]
				}
				if len(cloned[b].Fitness) > 0 {
					fitB = cloned[b].Fitness[0]
				}
				return fitA > fitB // descending order
			})
			for r := 0; r < effRate; r++ {
				selected[r] = NewIndividual[Env, State](cloned[r].Genome.Copy())
			}
		} else {
			// Random
			for r := 0; r < effRate; r++ {
				selected[r] = NewIndividual[Env, State](island.Population[rand.Intn(popSize)].Genome.Copy())
			}
		}
		migrants[i] = selected
	}

	// 2. Distribute migrants according to topology and replace in target
	for i := 0; i < numIslands; i++ {
		var targetIdx int
		if m.Config.MigrationTopology == TopologyRing {
			targetIdx = (i + 1) % numIslands
		} else {
			// Random target != current
			targetIdx = rand.Intn(numIslands - 1)
			if targetIdx >= i {
				targetIdx++
			}
		}

		targetIsland := m.Islands[targetIdx]
		targetPopSize := len(targetIsland.Population)
		if targetPopSize == 0 {
			continue
		}

		selectedMigrants := migrants[i]
		effRate := len(selectedMigrants)
		if effRate > targetPopSize {
			effRate = targetPopSize
		}

		if m.Config.MigrationPolicy == PolicyBestReplaceWorst {
			// Sort target population so we replace the worst
			sort.Slice(targetIsland.Population, func(a, b int) bool {
				fitA := 0.0
				fitB := 0.0
				if len(targetIsland.Population[a].Fitness) > 0 {
					fitA = targetIsland.Population[a].Fitness[0]
				}
				if len(targetIsland.Population[b].Fitness) > 0 {
					fitB = targetIsland.Population[b].Fitness[0]
				}
				return fitA > fitB // descending order
			})
			// Replace starting from the end (lowest fitness)
			for r := 0; r < effRate; r++ {
				replaceIdx := targetPopSize - 1 - r
				targetIsland.Population[replaceIdx] = selectedMigrants[r]
			}
		} else {
			// Random replacement
			for r := 0; r < effRate; r++ {
				replaceIdx := rand.Intn(targetPopSize)
				targetIsland.Population[replaceIdx] = selectedMigrants[r]
			}
		}
	}
}
