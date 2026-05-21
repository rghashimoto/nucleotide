package nucleotide

import (
	"testing"
)

func TestGenericTournamentSelector(t *testing.T) {
	pop := Population[TestEnv, struct{}]{
		{Fitness: 10},
		{Fitness: 20},
		{Fitness: 30},
	}
	s := GenericTournamentSelector[TestEnv, struct{}]{Size: 2}
	selected := s.SelectTyped(pop)
	if selected == nil {
		t.Fatal("Selected individual is nil")
	}
	
	// Test interface implementation
	var is Selector = s
	selInterface := is.Select(pop)
	if selInterface.(*Individual[TestEnv, struct{}]) == nil {
		t.Error("Interface Select failed")
	}
}

func TestProbabilisticTournamentSelector(t *testing.T) {
	pop := Population[TestEnv, struct{}]{
		{Fitness: 10.0, Genome: BitGenome{true}},
		{Fitness: 5.0, Genome: BitGenome{true}},
		{Fitness: 1.0, Genome: BitGenome{true}},
	}

	// We use Unique: true to ensure all 3 individuals are in the tournament,
	// allowing us to isolate probabilistic behavior directly.
	sHigh := NewProbabilisticTournamentSelector[TestEnv, struct{}](3, 0.999)
	sHigh.Unique = true

	sLow := NewProbabilisticTournamentSelector[TestEnv, struct{}](3, 0.001)
	sLow.Unique = true

	bestWins := 0
	for i := 0; i < 100; i++ {
		sel := sHigh.SelectTyped(pop)
		if sel.Fitness == 10.0 {
			bestWins++
		}
	}
	if bestWins < 90 {
		t.Errorf("Expected high fitness individual to win most of the time with P=0.999, got %d wins", bestWins)
	}

	worstWins := 0
	for i := 0; i < 100; i++ {
		sel := sLow.SelectTyped(pop)
		if sel.Fitness == 1.0 {
			worstWins++
		}
	}
	if worstWins < 90 {
		t.Errorf("Expected lowest fitness individual to win most of the time with P=0.001, got %d wins", worstWins)
	}
}

func TestAdaptiveTournamentSelector(t *testing.T) {
	pop := Population[TestEnv, struct{}]{
		{Fitness: 1.0, Genome: BitGenome{true}},
		{Fitness: 2.0, Genome: BitGenome{true}},
		{Fitness: 3.0, Genome: BitGenome{true}},
		{Fitness: 4.0, Genome: BitGenome{true}},
		{Fitness: 5.0, Genome: BitGenome{true}},
	}

	progress := 0.0
	progressFunc := func() float64 {
		return progress
	}

	s := NewAdaptiveTournamentSelector[TestEnv, struct{}](1, 5, progressFunc)
	s.Unique = true // Ensure deterministic selection when size matches population size

	// At start progress = 0.0, effective size is 1.
	// Best (5.0) should not win 100% of the time.
	progress = 0.0
	fiveWinsStart := 0
	for i := 0; i < 100; i++ {
		if s.SelectTyped(pop).Fitness == 5.0 {
			fiveWinsStart++
		}
	}
	if fiveWinsStart == 100 {
		t.Errorf("Size should be 1 at start progress, but got 100%% wins for the best individual")
	}

	// At end progress = 1.0, effective size is 5.
	// Best (5.0) must win 100% of the time.
	progress = 1.0
	fiveWinsEnd := 0
	for i := 0; i < 100; i++ {
		if s.SelectTyped(pop).Fitness == 5.0 {
			fiveWinsEnd++
		}
	}
	if fiveWinsEnd != 100 {
		t.Errorf("Expected best individual to win 100%% of the time at end progress (size 5), got %d wins", fiveWinsEnd)
	}
}

func TestNichingTournamentSelector(t *testing.T) {
	// Two identical individuals (BitGenome{true, true}) with fitness 10.0
	// One different individual (BitGenome{false, false}) with fitness 8.0
	pop := Population[TestEnv, struct{}]{
		{Fitness: 10.0, Genome: BitGenome{true, true}},
		{Fitness: 10.0, Genome: BitGenome{true, true}},
		{Fitness: 8.0, Genome: BitGenome{false, false}},
	}

	// Without niching, the best (10.0) always wins
	sNoNiche := GenericTournamentSelector[TestEnv, struct{}]{Size: 3}
	sNoNiche.Unique = true
	if sNoNiche.SelectTyped(pop).Fitness != 10.0 {
		t.Errorf("Expected 10.0 without niching")
	}

	// With niching (SigmaShare = 1.0)
	// Hamming distance between identical is 0.0, which is < SigmaShare (1.0).
	// They share and penalize each other: NicheCount = 1 + 1 = 2. AdjustedFit = 10.0 / 2.0 = 5.0.
	// The different individual has distance 1.0 from others, so it doesn't share.
	// AdjustedFit of different = 8.0 / 1.0 = 8.0.
	// Since 8.0 > 5.0, the different individual wins.
	sNiche := NewNichingTournamentSelector[TestEnv, struct{}](3, 1.0, nil)
	sNiche.Unique = true
	sel := sNiche.SelectTyped(pop)
	if sel.Fitness != 8.0 {
		t.Errorf("Expected different individual with fitness 8.0 to win due to niche penalty of duplicate individuals, got %f", sel.Fitness)
	}
}

func TestUniqueTournamentSelector(t *testing.T) {
	pop := Population[TestEnv, struct{}]{
		{Fitness: 10.0, Genome: BitGenome{true}},
		{Fitness: 5.0, Genome: BitGenome{true}},
	}

	// With replacement (size 2), worst individual (5.0) can occasionally win
	sWithRep := GenericTournamentSelector[TestEnv, struct{}]{Size: 2}
	worstWinsWithRep := 0
	for i := 0; i < 100; i++ {
		if sWithRep.SelectTyped(pop).Fitness == 5.0 {
			worstWinsWithRep++
		}
	}

	// Without replacement (size 2), worst individual can NEVER win (since best is always present in a size-2 tournament of a size-2 population)
	sUnique := NewUniqueTournamentSelector[TestEnv, struct{}](2)
	worstWinsUnique := 0
	for i := 0; i < 100; i++ {
		if sUnique.SelectTyped(pop).Fitness == 5.0 {
			worstWinsUnique++
		}
	}

	if worstWinsUnique != 0 {
		t.Errorf("Expected 0 wins for worst individual in unique tournament of size 2, got %d wins", worstWinsUnique)
	}
	
	t.Logf("Worst individual won %d times with replacement, and %d times without replacement", worstWinsWithRep, worstWinsUnique)
}

func TestRouletteWheelSelector(t *testing.T) {
	pop := Population[TestEnv, struct{}]{
		{Fitness: 10.0, Genome: BitGenome{true}},
		{Fitness: 20.0, Genome: BitGenome{true}},
		{Fitness: 30.0, Genome: BitGenome{true}},
	}
	s := RouletteWheelSelector[TestEnv, struct{}]{AutoShift: true}
	for i := 0; i < 100; i++ {
		sel := s.SelectTyped(pop)
		if sel == nil {
			t.Fatal("RouletteWheelSelector returned nil selection")
		}
	}

	negPop := Population[TestEnv, struct{}]{
		{Fitness: -10.0, Genome: BitGenome{true}},
		{Fitness: -5.0, Genome: BitGenome{true}},
	}
	for i := 0; i < 50; i++ {
		sel := s.SelectTyped(negPop)
		if sel == nil {
			t.Fatal("RouletteWheelSelector failed to select with negative fitness auto-shifting")
		}
	}
}

func TestStochasticUniversalSamplingSelector(t *testing.T) {
	pop := Population[TestEnv, struct{}]{
		{Fitness: 10.0, Genome: BitGenome{true}},
		{Fitness: 20.0, Genome: BitGenome{true}},
		{Fitness: 30.0, Genome: BitGenome{true}},
	}
	s := StochasticUniversalSamplingSelector[TestEnv, struct{}]{AutoShift: true}
	
	for i := 0; i < 100; i++ {
		sel := s.SelectTyped(pop)
		if sel == nil {
			t.Fatal("StochasticUniversalSamplingSelector returned nil selection")
		}
	}
}

func TestRankSelector(t *testing.T) {
	pop := Population[TestEnv, struct{}]{
		{Fitness: 100.0, Genome: BitGenome{true}},
		{Fitness: 1.0, Genome: BitGenome{true}},
	}
	s := RankSelector[TestEnv, struct{}]{SelectionPressure: 2.0}
	
	bestCount := 0
	for i := 0; i < 100; i++ {
		sel := s.SelectTyped(pop)
		if sel.Fitness == 100.0 {
			bestCount++
		}
	}
	if bestCount != 100 {
		t.Errorf("Expected best individual to be selected 100 times under linear rank selection with SP 2.0, got %d", bestCount)
	}
}

func TestBoltzmannSelector(t *testing.T) {
	pop := Population[TestEnv, struct{}]{
		{Fitness: 100.0, Genome: BitGenome{true}},
		{Fitness: 1.0, Genome: BitGenome{true}},
	}
	
	sCold := BoltzmannSelector[TestEnv, struct{}]{Temperature: 0.0001}
	coldBestWins := 0
	for i := 0; i < 100; i++ {
		if sCold.SelectTyped(pop).Fitness == 100.0 {
			coldBestWins++
		}
	}
	if coldBestWins != 100 {
		t.Errorf("Expected cold Boltzmann selection to always select best, got %d wins", coldBestWins)
	}
}

func TestGenericTournamentSelector_AdaptiveDiversity(t *testing.T) {
	uniformPop := Population[TestEnv, struct{}]{
		{Fitness: 10.0, Genome: BitGenome{true}},
		{Fitness: 10.0, Genome: BitGenome{true}},
		{Fitness: 10.0, Genome: BitGenome{true}},
	}

	s := GenericTournamentSelector[TestEnv, struct{}]{
		Size:              3,
		MinSize:           2,
		MaxSize:           4,
		AdaptiveDiversity: true,
	}

	sel := s.SelectTyped(uniformPop)
	if sel == nil {
		t.Error("AdaptiveDiversity selection failed")
	}
}

func TestGenericTournamentSelector_AgeBias(t *testing.T) {
	pop := Population[TestEnv, struct{}]{
		{Fitness: 10.0, Genome: BitGenome{true}, Age: 0},
		{Fitness: 10.0, Genome: BitGenome{true}, Age: 10},
	}

	s := GenericTournamentSelector[TestEnv, struct{}]{
		Size:    2,
		Unique:  true,
		AgeBias: 1.0,
	}

	for i := 0; i < 100; i++ {
		sel := s.SelectTyped(pop)
		if sel.Age != 0 {
			t.Errorf("Expected young individual to win due to age penalty, but got age %d", sel.Age)
			break
		}
	}
}

func TestGenericTournamentSelector_HallOfFame(t *testing.T) {
	pop := Population[TestEnv, struct{}]{
		{Fitness: 1.0, Genome: BitGenome{true}},
	}
	hof := Population[TestEnv, struct{}]{
		{Fitness: 99.0, Genome: BitGenome{true}},
	}

	s := GenericTournamentSelector[TestEnv, struct{}]{
		Size:                  2,
		HallOfFame:            &hof,
		HallOfFameProbability: 1.0,
	}

	sel := s.SelectTyped(pop)
	if sel.Fitness != 99.0 && sel.Fitness != 1.0 {
		t.Errorf("Expected individual with fitness 99.0 or 1.0, got %f", sel.Fitness)
	}
}

func TestGenericTournamentSelector_SelfAdaptive(t *testing.T) {
	p1 := Individual[TestEnv, CustomSelfAdaptiveState]{
		Fitness: 1.0,
		Genome:  BitGenome{true},
		State:   CustomSelfAdaptiveState{PreferredK: 1},
	}

	p2 := Individual[TestEnv, CustomSelfAdaptiveState]{
		Fitness: 10.0,
		Genome:  BitGenome{true},
		State:   CustomSelfAdaptiveState{PreferredK: 1},
	}

	pop := Population[TestEnv, CustomSelfAdaptiveState]{
		&p1,
		&p2,
	}

	s := GenericTournamentSelector[TestEnv, CustomSelfAdaptiveState]{
		Size:         2,
		SelfAdaptive: true,
	}

	wins := 0
	for i := 0; i < 100; i++ {
		sel := s.SelectTyped(pop)
		if sel.Fitness == 1.0 {
			wins++
		}
	}
	if wins == 0 {
		t.Error("SelfAdaptive with size 1 failed to allow p1 to win")
	}
}
