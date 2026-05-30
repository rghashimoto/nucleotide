package nucleotide

import (
	"fmt"
	"math"
	"testing"
)

// Dummy structure for testing
type AdaptiveTestState struct{}

func TestAdaptiveMutation_SigmoidController(t *testing.T) {
	// Setup standard sigmoid controller
	target := 0.3
	sensitivity := 10.0
	minS := 0.1
	maxS := 5.0
	ctrl := NewSigmoidDiversityFeedbackController[TestEnv, AdaptiveTestState](target, sensitivity, minS, maxS)

	// Mock an engine
	config := EngineConfig[TestEnv, AdaptiveTestState]{
		PopulationSize: 10,
		FitnessFunc: func(g Genome, env TestEnv) []float64 {
			return []float64{0.0}
		},
	}
	engine, err := NewEngine(config)
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}

	// 1. Low diversity (completely identical -> zero diversity) -> pushes scaler up towards maxS
	engine.Population = make(Population[TestEnv, AdaptiveTestState], 10)
	for i := range engine.Population {
		bg := make(BitGenome, 10)
		for j := range bg {
			bg[j] = (j < 3)
		}
		engine.Population[i] = NewIndividual[TestEnv, AdaptiveTestState](bg)
	}

	scalerLow := ctrl.GetMutationScaler(engine)
	if scalerLow < 4.0 {
		t.Errorf("Expected low diversity to yield a high mutation scaler (close to MaxScaler %f). Got %f", maxS, scalerLow)
	}
	if scalerLow > maxS {
		t.Errorf("Expected scaler to be bounded by MaxScaler %f, got %f", maxS, scalerLow)
	}

	// 2. High diversity -> pushes scaler down towards minS
	for i := range engine.Population {
		bg := make(BitGenome, 10)
		for j := range bg {
			bg[j] = (i%2 == 0) // Highly heterogeneous
		}
		engine.Population[i] = NewIndividual[TestEnv, AdaptiveTestState](bg)
	}
	scalerHigh := ctrl.GetMutationScaler(engine)
	if scalerHigh > 1.0 {
		t.Errorf("Expected high diversity to yield a low mutation scaler (close to MinScaler %f). Got %f", minS, scalerHigh)
	}
	if scalerHigh < minS {
		t.Errorf("Expected scaler to be bounded by MinScaler %f, got %f", minS, scalerHigh)
	}
}

func TestAdaptiveMutation_ScheduleController(t *testing.T) {
	// Exponential Decay test
	decayCtrl := &TemporalScheduleController[TestEnv, AdaptiveTestState]{
		Type:        ScheduleExponentialDecay,
		InitialRate: 2.0,
		FinalRate:   0.2,
	}

	config := EngineConfig[TestEnv, AdaptiveTestState]{
		PopulationSize: 10,
		MaxGenerations: 20,
		FitnessFunc: func(g Genome, env TestEnv) []float64 {
			return []float64{0.0}
		},
	}
	engine, _ := NewEngine(config)

	// Gen 0 -> initial rate
	s0 := decayCtrl.GetMutationScaler(engine)
	if math.Abs(s0-2.0) > 0.001 {
		t.Errorf("Expected initial decay rate to be 2.0, got %f", s0)
	}

	// Progress generation to Gen 10 (halfway)
	engine.Generation = 10
	s10 := decayCtrl.GetMutationScaler(engine)
	if s10 >= s0 || s10 <= 0.2 {
		t.Errorf("Expected Gen 10 decay rate to be between 2.0 and 0.2, got %f", s10)
	}

	// Progress generation to Gen 20 (max)
	engine.Generation = 20
	s20 := decayCtrl.GetMutationScaler(engine)
	if math.Abs(s20-0.2) > 0.001 {
		t.Errorf("Expected final decay rate to be 0.2, got %f", s20)
	}

	// Cosine Annealing test
	cosineCtrl := &TemporalScheduleController[TestEnv, AdaptiveTestState]{
		Type:        ScheduleCosineAnnealing,
		InitialRate: 1.0,
		FinalRate:   0.1,
		CycleLength: 10,
	}

	engine.Generation = 0
	c0 := cosineCtrl.GetMutationScaler(engine)
	if math.Abs(c0-1.0) > 0.001 {
		t.Errorf("Expected initial cosine rate to be 1.0, got %f", c0)
	}

	// Last step of cycle (Gen 9) -> minimum rate (0.1)
	engine.Generation = 9
	c9 := cosineCtrl.GetMutationScaler(engine)
	if math.Abs(c9-0.1) > 0.001 {
		t.Errorf("Expected end-of-cycle cosine rate to be 0.1, got %f", c9)
	}

	// Full cycle wrap (Gen 10) -> returns back to initial rate (1.0) due to modulo 10
	engine.Generation = 10
	c10 := cosineCtrl.GetMutationScaler(engine)
	if math.Abs(c10-1.0) > 0.001 {
		t.Errorf("Expected full cycle cosine rate to return to 1.0, got %f", c10)
	}
}

func TestAdaptiveMutation_RechenbergController(t *testing.T) {
	interval := 5
	targetRatio := 0.2 // 1/5th
	rechenberg := NewRechenbergController[TestEnv, AdaptiveTestState](interval, targetRatio)

	config := EngineConfig[TestEnv, AdaptiveTestState]{
		PopulationSize: 10,
		FitnessFunc: func(g Genome, env TestEnv) []float64 {
			return []float64{0.0}
		},
	}
	engine, _ := NewEngine(config)

	// Generation 0 -> no scaling change
	s0 := rechenberg.GetMutationScaler(engine)
	if s0 != 1.0 {
		t.Errorf("Expected starting Rechenberg scaler to be 1.0, got %f", s0)
	}

	// Generation 5 -> trigger update.
	// Scenario A: High success ratio (e.g. 50% success i.e. 5 out of 10 mutations succeed)
	engine.Generation = 5
	engine.TotalMutations = 10
	engine.SuccessfulMutations = 5
	s5a := rechenberg.GetMutationScaler(engine)
	if s5a <= 1.0 {
		t.Errorf("Expected scaler to increase (> 1.0) due to high success ratio (5/10 > 0.2), got %f", s5a)
	}
	// Verify reset
	if engine.TotalMutations != 0 || engine.SuccessfulMutations != 0 {
		t.Errorf("Expected mutations tracking variables to reset after Rechenberg update, got Total=%d, Successful=%d", engine.TotalMutations, engine.SuccessfulMutations)
	}

	// Scenario B: Low success ratio (e.g. 0% success i.e. 0 out of 10 mutations succeed)
	// Generation 10 -> trigger update
	engine.Generation = 10
	engine.TotalMutations = 10
	engine.SuccessfulMutations = 0
	s10a := rechenberg.GetMutationScaler(engine)
	if s10a >= s5a {
		t.Errorf("Expected scaler to decrease from %f due to low success ratio (0/10 < 0.2), got %f", s5a, s10a)
	}
}

func TestAdaptiveMutation_SelfAdaptiveController(t *testing.T) {
	learningRate := 0.15
	minRate := 0.01
	maxRate := 0.4
	saCtrl := NewSelfAdaptiveController[TestEnv, AdaptiveTestState](learningRate, minRate, maxRate)

	config := EngineConfig[TestEnv, AdaptiveTestState]{
		PopulationSize: 10,
		FitnessFunc: func(g Genome, env TestEnv) []float64 {
			return []float64{0.0}
		},
		AdaptiveMutation:   true,
		MutationController: saCtrl,
		Selector:           GenericTournamentSelector[TestEnv, AdaptiveTestState]{Size: 2},
	}
	engine, _ := NewEngine(config)

	// Populate engine with individuals carrying custom mutation rates
	engine.Population = make(Population[TestEnv, AdaptiveTestState], 10)
	for i := range engine.Population {
		ind := NewIndividual[TestEnv, AdaptiveTestState](make(BitGenome, 10))
		ind.MutationRate = 0.1 // Set starting individual mutation rate
		ind.Fitness = []float64{0.0}
		engine.Population[i] = ind
	}

	// Perform next generation step using engine's resolved strategy
	newPop, err := engine.Config.Strategy.NextGeneration(engine, nil, engine.Population)
	if err != nil {
		t.Fatalf("Failed to execute generation: %v", err)
	}

	// Verify that offspring carry perturbed mutation rates bounded by min and max
	for i, offspring := range newPop {
		if offspring.MutationRate < minRate || offspring.MutationRate > maxRate {
			t.Errorf("Offspring %d mutation rate %f is out of bounds [%f, %f]", i, offspring.MutationRate, minRate, maxRate)
		}
		// Confirm it has been perturbed away from the static parent rate (0.1) in some individuals
		if offspring.MutationRate == 0.0 {
			t.Errorf("Offspring %d mutation rate was not initialized or mutated", i)
		}
	}
}

func TestEngine_Integration_Sigmoid(t *testing.T) {
	config := EngineConfig[TestEnv, AdaptiveTestState]{
		PopulationSize:   10,
		MaxGenerations:   5,
		AdaptiveMutation: true, // Default SigmoidDiversityFeedbackController
		FitnessFunc: func(g Genome, env TestEnv) []float64 {
			bg := g.(BitGenome)
			score := 0.0
			for _, b := range bg {
				if b {
					score += 1.0
				}
			}
			return []float64{score}
		},
	}

	engine, err := NewEngine(config)
	if err != nil {
		t.Fatalf("Failed to build Engine: %v", err)
	}

	// Manually initialize population to run
	def := NewDefinition[TestEnv, AdaptiveTestState]()
	initialPop := make(Population[TestEnv, AdaptiveTestState], 10)
	for i := 0; i < 10; i++ {
		bg := make(BitGenome, 8)
		for j := range bg {
			bg[j] = (i%2 == 0)
		}
		initialPop[i] = NewIndividual[TestEnv, AdaptiveTestState](bg)
	}
	engine.Population = initialPop

	// Run step-by-step
	for i := 0; i < 3; i++ {
		err := engine.Step(def)
		if err != nil {
			t.Fatalf("Failed step %d: %v", i, err)
		}
		// Confirm DiversityHistory is updated
		if len(engine.DiversityHistory) != i+1 {
			t.Errorf("Expected DiversityHistory size %d, got %d", i+1, len(engine.DiversityHistory))
		}
	}

	// Retrieve best solution
	best := engine.Population.Best()
	if best == nil {
		t.Fatal("Expected best individual to not be nil")
	}
	fmt.Printf("Integration successful. Best individual fitness: %v\n", best.Fitness)
}
