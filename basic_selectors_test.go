package nucleotide

import (
	"testing"
)

func TestRouletteWheelSelector(t *testing.T) {
	pop := Population[TestEnv, struct{}]{
		{Fitness: []float64{10.0}, Genome: BitGenome{true}},
		{Fitness: []float64{20.0}, Genome: BitGenome{true}},
		{Fitness: []float64{30.0}, Genome: BitGenome{true}},
	}
	s := RouletteWheelSelector[TestEnv, struct{}]{AutoShift: true}
	for i := 0; i < 100; i++ {
		sel := s.SelectTyped(pop)
		if sel == nil {
			t.Fatal("RouletteWheelSelector returned nil selection")
		}
	}

	negPop := Population[TestEnv, struct{}]{
		{Fitness: []float64{-10.0}, Genome: BitGenome{true}},
		{Fitness: []float64{-5.0}, Genome: BitGenome{true}},
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
		{Fitness: []float64{10.0}, Genome: BitGenome{true}},
		{Fitness: []float64{20.0}, Genome: BitGenome{true}},
		{Fitness: []float64{30.0}, Genome: BitGenome{true}},
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
		{Fitness: []float64{100.0}, Genome: BitGenome{true}},
		{Fitness: []float64{1.0}, Genome: BitGenome{true}},
	}
	s := RankSelector[TestEnv, struct{}]{SelectionPressure: 2.0}
	
	bestCount := 0
	for i := 0; i < 100; i++ {
		sel := s.SelectTyped(pop)
		if len(sel.Fitness) > 0 && sel.Fitness[0] == 100.0 {
			bestCount++
		}
	}
	if bestCount != 100 {
		t.Errorf("Expected best individual to be selected 100 times under linear rank selection with SP 2.0, got %d", bestCount)
	}
}

func TestBoltzmannSelector(t *testing.T) {
	pop := Population[TestEnv, struct{}]{
		{Fitness: []float64{100.0}, Genome: BitGenome{true}},
		{Fitness: []float64{1.0}, Genome: BitGenome{true}},
	}
	
	sCold := BoltzmannSelector[TestEnv, struct{}]{Temperature: 0.0001}
	coldBestWins := 0
	for i := 0; i < 100; i++ {
		sel := sCold.SelectTyped(pop)
		if len(sel.Fitness) > 0 && sel.Fitness[0] == 100.0 {
			coldBestWins++
		}
	}
	if coldBestWins != 100 {
		t.Errorf("Expected cold Boltzmann selection to always select best, got %d wins", coldBestWins)
	}
}
