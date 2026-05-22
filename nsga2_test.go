package nucleotide

import (
	"math"
	"testing"
)

func TestFastNondominatedSorting(t *testing.T) {
	// Let's create 4 individuals with conflicting objectives
	// Objectives are: Maximize, Maximize
	pop := Population[TestEnv, struct{}]{
		{Fitness: []float64{10.0, 10.0}}, // Ind 0
		{Fitness: []float64{5.0, 5.0}},   // Ind 1
		{Fitness: []float64{15.0, 5.0}},  // Ind 2
		{Fitness: []float64{5.0, 15.0}},  // Ind 3
	}

	directions := []ObjectiveDirection{Maximize, Maximize}
	fronts := fastNondominatedSort(pop, directions)

	// Assertions based on mathematical dominance analysis:
	// Front 0 (Rank 0) must contain Ind 0, Ind 2, Ind 3 since they are incomparable and not dominated.
	// Front 1 (Rank 1) must contain Ind 1 since it is dominated by all others.
	if len(fronts) < 2 {
		t.Fatalf("Expected at least 2 fronts, got %d", len(fronts))
	}

	// Verify Front 0
	front0 := fronts[0]
	if len(front0) != 3 {
		t.Errorf("Expected Front 0 to have 3 individuals, got %d", len(front0))
	}

	hasInd0 := false
	hasInd2 := false
	hasInd3 := false
	for _, idx := range front0 {
		if idx == 0 {
			hasInd0 = true
		}
		if idx == 2 {
			hasInd2 = true
		}
		if idx == 3 {
			hasInd3 = true
		}
	}
	if !hasInd0 || !hasInd2 || !hasInd3 {
		t.Errorf("Front 0 has incorrect elements: hasInd0=%t, hasInd2=%t, hasInd3=%t", hasInd0, hasInd2, hasInd3)
	}

	// Verify Front 1
	front1 := fronts[1]
	if len(front1) != 1 || front1[0] != 1 {
		t.Errorf("Expected Front 1 to contain only Ind 1, got %v", front1)
	}

	// Verify ranks were correctly written to individuals
	if pop[0].Rank != 0 || pop[2].Rank != 0 || pop[3].Rank != 0 {
		t.Errorf("Expected rank 0 for Inds 0, 2, 3. Got: Ind0=%d, Ind2=%d, Ind3=%d", pop[0].Rank, pop[2].Rank, pop[3].Rank)
	}
	if pop[1].Rank != 1 {
		t.Errorf("Expected rank 1 for Ind 1, got %d", pop[1].Rank)
	}
}

func TestCrowdingDistanceCalculation(t *testing.T) {
	pop := Population[TestEnv, struct{}]{
		{Fitness: []float64{10.0, 10.0}}, // Ind 0
		{Fitness: []float64{15.0, 5.0}},  // Ind 1
		{Fitness: []float64{5.0, 15.0}},  // Ind 2
	}

	directions := []ObjectiveDirection{Maximize, Maximize}
	frontIndices := []int{0, 1, 2}

	calculateCrowdingDistances(pop, frontIndices, directions)

	// Math Check:
	// Sorted by Obj 0: Ind 2 (5), Ind 0 (10), Ind 1 (15)
	// Boundaries (Ind 2, Ind 1) get MaxFloat64
	// Intermediate (Ind 0) gets: (15 - 5) / (15 - 5) = 1.0
	// Sorted by Obj 1: Ind 1 (5), Ind 0 (10), Ind 2 (15)
	// Boundaries (Ind 1, Ind 2) get MaxFloat64
	// Intermediate (Ind 0) gets: + (15 - 5) / (15 - 5) = 1.0
	// Final CrowdingDistance of Ind 0 = 2.0

	if pop[1].CrowdingDistance != math.MaxFloat64 {
		t.Errorf("Expected boundary Ind 1 to have infinite distance, got %f", pop[1].CrowdingDistance)
	}
	if pop[2].CrowdingDistance != math.MaxFloat64 {
		t.Errorf("Expected boundary Ind 2 to have infinite distance, got %f", pop[2].CrowdingDistance)
	}
	if pop[0].CrowdingDistance != 2.0 {
		t.Errorf("Expected intermediate Ind 0 to have crowding distance 2.0, got %f", pop[0].CrowdingDistance)
	}
}

func TestNSGA2Generation(t *testing.T) {
	// Define a custom BitGenome population factory
	popFunc := func(def *Definition[TestEnv, struct{}], size int) Population[TestEnv, struct{}] {
		pop := make(Population[TestEnv, struct{}], size)
		for i := 0; i < size; i++ {
			// Alternate initial bits
			bg := make(BitGenome, 5)
			for j := 0; j < 5; j++ {
				bg[j] = (i+j)%2 == 0
			}
			pop[i] = NewIndividual[TestEnv, struct{}](bg)
		}
		return pop
	}

	config := EngineConfig[TestEnv, struct{}]{
		PopulationSize: 20,
		MaxGenerations: 5,
		FitnessFunc: func(g Genome, env TestEnv) []float64 {
			bg := g.(BitGenome)
			trues := 0.0
			for _, b := range bg {
				if b {
					trues++
				}
			}
			fols := float64(len(bg)) - trues
			return []float64{trues, fols}
		},
		Selector:            GenericTournamentSelector[TestEnv, struct{}]{Size: 3},
		ObjectiveDirections: []ObjectiveDirection{Maximize, Maximize},
		PopulationFunc:      popFunc,
	}

	engine, err := NewEngine[TestEnv, struct{}](config)
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}

	def := NewDefinition[TestEnv, struct{}]()
	best, err := engine.Run(def)
	if err != nil {
		t.Fatalf("Failed to run multi-objective optimization: %v", err)
	}

	if best == nil {
		t.Fatal("Expected best individual to not be nil")
	}

	frontier := engine.ParetoFrontier()
	if len(frontier) == 0 {
		t.Error("Expected non-empty Pareto frontier")
	}

	// All individuals in the frontier must have Rank == 0
	for _, ind := range frontier {
		if ind.Rank != 0 {
			t.Errorf("Pareto frontier contains individual with non-zero rank: %d", ind.Rank)
		}
	}
}
