package nucleotide

import (
	"testing"
)

// Reuse TestEnv and AdaptiveTestState from previous tests
type OperatorsTestState struct{}

func TestSelectors_RouletteWheel(t *testing.T) {
	// 1. Positive fitness verification
	pop := make(Population[TestEnv, OperatorsTestState], 3)
	pop[0] = &Individual[TestEnv, OperatorsTestState]{Fitness: []float64{10.0}}
	pop[1] = &Individual[TestEnv, OperatorsTestState]{Fitness: []float64{20.0}}
	pop[2] = &Individual[TestEnv, OperatorsTestState]{Fitness: []float64{70.0}}

	selector := RouletteWheelSelector[TestEnv, OperatorsTestState]{}
	counts := make(map[float64]int)
	for i := 0; i < 1000; i++ {
		selected := selector.SelectTyped(pop)
		counts[selected.Fitness[0]]++
	}

	// Verify that higher fitness is selected more often
	if counts[70.0] < counts[20.0] || counts[20.0] < counts[10.0] {
		t.Errorf("RouletteWheel selection proportions skewed: counts[70]=%d, counts[20]=%d, counts[10]=%d", counts[70.0], counts[20.0], counts[10.0])
	}

	// 2. Negative fitness linear shifting verification
	negPop := make(Population[TestEnv, OperatorsTestState], 3)
	negPop[0] = &Individual[TestEnv, OperatorsTestState]{Fitness: []float64{-10.0}}
	negPop[1] = &Individual[TestEnv, OperatorsTestState]{Fitness: []float64{-5.0}}
	negPop[2] = &Individual[TestEnv, OperatorsTestState]{Fitness: []float64{-1.0}}

	// Shifted values should be:
	// minFit = -10.0, offset = 10.1
	// f0 = -10.0 + 10.1 = 0.1
	// f1 = -5.0 + 10.1 = 5.1
	// f2 = -1.0 + 10.1 = 9.1
	// Verify that the linear shifting completes without panic and resolves selection
	countsNeg := make(map[float64]int)
	for i := 0; i < 1000; i++ {
		selected := selector.SelectTyped(negPop)
		countsNeg[selected.Fitness[0]]++
	}
	if countsNeg[-1.0] < countsNeg[-5.0] || countsNeg[-5.0] < countsNeg[-10.0] {
		t.Errorf("Linear shifting selection failed: counts[-1]=%d, counts[-5]=%d, counts[-10]=%d", countsNeg[-1.0], countsNeg[-5.0], countsNeg[-10.0])
	}

	// 3. Multi-objective (NSGA-II) synthetic scalar fitness verification
	moPop := make(Population[TestEnv, OperatorsTestState], 3)
	// Lower rank (0) is better than higher rank (1)
	moPop[0] = &Individual[TestEnv, OperatorsTestState]{Fitness: []float64{10.0, 20.0}, Rank: 0, CrowdingDistance: 2.0} // Evolved best
	moPop[1] = &Individual[TestEnv, OperatorsTestState]{Fitness: []float64{8.0, 15.0}, Rank: 0, CrowdingDistance: 0.5}  // Rank 0, lower crowding
	moPop[2] = &Individual[TestEnv, OperatorsTestState]{Fitness: []float64{5.0, 10.0}, Rank: 1, CrowdingDistance: 1.0}  // Rank 1
	// Synthetic fitness:
	// ind0: 1 / 1 + 2 / 3 = 1.0 + 0.667 = 1.667
	// ind1: 1 / 1 + 0.5 / 1.5 = 1.0 + 0.333 = 1.333
	// ind2: 1 / 2 + 1.0 / 2.0 = 0.5 + 0.5 = 1.0

	countsMO := make(map[float64]int)
	for i := 0; i < 1000; i++ {
		selected := selector.SelectTyped(moPop)
		countsMO[selected.Fitness[0]]++
	}
	if countsMO[10.0] < countsMO[8.0] || countsMO[8.0] < countsMO[5.0] {
		t.Errorf("NSGA-II synthetic selection failed: counts[10.0]=%d, counts[8.0]=%d, counts[5.0]=%d", countsMO[10.0], countsMO[8.0], countsMO[5.0])
	}
}

func TestSelectors_SUS(t *testing.T) {
	pop := make(Population[TestEnv, OperatorsTestState], 4)
	pop[0] = &Individual[TestEnv, OperatorsTestState]{Fitness: []float64{10.0}}
	pop[1] = &Individual[TestEnv, OperatorsTestState]{Fitness: []float64{20.0}}
	pop[2] = &Individual[TestEnv, OperatorsTestState]{Fitness: []float64{30.0}}
	pop[3] = &Individual[TestEnv, OperatorsTestState]{Fitness: []float64{40.0}}

	// Setup SUS
	sus := &StochasticUniversalSamplingSelector[TestEnv, OperatorsTestState]{}

	// Verify that selections are made without errors and fill/deplete cache
	counts := make(map[float64]int)
	for i := 0; i < 2000; i++ {
		selected := sus.SelectTyped(pop)
		counts[selected.Fitness[0]]++
	}

	if counts[40.0] < counts[20.0] || counts[30.0] < counts[10.0] {
		t.Errorf("SUS selection failed: counts[40]=%d, counts[30]=%d, counts[20]=%d, counts[10]=%d", counts[40.0], counts[30.0], counts[20.0], counts[10.0])
	}
}

func TestSelectors_Ranking(t *testing.T) {
	pop := make(Population[TestEnv, OperatorsTestState], 3)
	pop[0] = &Individual[TestEnv, OperatorsTestState]{Fitness: []float64{10.0}}
	pop[1] = &Individual[TestEnv, OperatorsTestState]{Fitness: []float64{50.0}}
	pop[2] = &Individual[TestEnv, OperatorsTestState]{Fitness: []float64{100.0}}

	// 1. Linear Rank Selector
	linearRank := LinearRankSelector[TestEnv, OperatorsTestState]{SelectionPressure: 1.5}
	countsL := make(map[float64]int)
	for i := 0; i < 1000; i++ {
		selected := linearRank.SelectTyped(pop)
		countsL[selected.Fitness[0]]++
	}
	if countsL[100.0] < countsL[50.0] || countsL[50.0] < countsL[10.0] {
		t.Errorf("Linear Rank selection failed: counts[100]=%d, counts[50]=%d, counts[10]=%d", countsL[100.0], countsL[50.0], countsL[10.0])
	}

	// 2. Exponential Rank Selector
	expRank := ExponentialRankSelector[TestEnv, OperatorsTestState]{C: 0.5}
	countsE := make(map[float64]int)
	for i := 0; i < 1000; i++ {
		selected := expRank.SelectTyped(pop)
		countsE[selected.Fitness[0]]++
	}
	if countsE[100.0] < countsE[50.0] || countsE[50.0] < countsE[10.0] {
		t.Errorf("Exponential Rank selection failed: counts[100]=%d, counts[50]=%d, counts[10]=%d", countsE[100.0], countsE[50.0], countsE[10.0])
	}
}

func TestCrossovers_OX(t *testing.T) {
	// Order Crossover verification
	p1 := SequenceGenome{1, 2, 3, 4, 5, 6, 7, 8}
	p2 := SequenceGenome{5, 6, 7, 8, 1, 2, 3, 4}

	ox := OrderCrossover{}
	off1, off2 := ox.Crossover(p1, p2)

	seq1 := off1.(SequenceGenome)
	seq2 := off2.(SequenceGenome)

	// Verify lengths
	if len(seq1) != 8 || len(seq2) != 8 {
		t.Fatalf("Expected offspring length 8, got %d and %d", len(seq1), len(seq2))
	}

	// Verify uniqueness of elements (valid permutation)
	verifyPermutation := func(seq SequenceGenome) {
		seen := make(map[int]bool)
		for _, v := range seq {
			if v < 1 || v > 8 {
				t.Errorf("Value %d is out of bounds [1, 8]", v)
			}
			if seen[v] {
				t.Errorf("Duplicate value %d found in permutation %v", v, seq)
			}
			seen[v] = true
		}
	}

	verifyPermutation(seq1)
	verifyPermutation(seq2)
}

func TestCrossovers_CX(t *testing.T) {
	// Cycle Crossover verification
	p1 := SequenceGenome{1, 2, 3, 4, 5, 6, 7, 8}
	p2 := SequenceGenome{8, 5, 2, 1, 3, 6, 7, 4}

	cx := CycleCrossover{}
	off1, off2 := cx.Crossover(p1, p2)

	seq1 := off1.(SequenceGenome)
	seq2 := off2.(SequenceGenome)

	// Verify length
	if len(seq1) != 8 || len(seq2) != 8 {
		t.Fatalf("Expected offspring length 8, got %d and %d", len(seq1), len(seq2))
	}

	// Verify uniqueness
	verifyPermutation := func(seq SequenceGenome) {
		seen := make(map[int]bool)
		for _, v := range seq {
			if seen[v] {
				t.Errorf("Duplicate value %d found in permutation %v", v, seq)
			}
			seen[v] = true
		}
	}
	verifyPermutation(seq1)
	verifyPermutation(seq2)
}

func TestCrossovers_ERX(t *testing.T) {
	// Edge Recombination Crossover verification
	p1 := SequenceGenome{1, 2, 3, 4, 5}
	p2 := SequenceGenome{2, 4, 1, 3, 5}

	erx := EdgeRecombinationCrossover{}
	off1, off2 := erx.Crossover(p1, p2)

	seq1 := off1.(SequenceGenome)
	seq2 := off2.(SequenceGenome)

	// Verify lengths
	if len(seq1) != 5 || len(seq2) != 5 {
		t.Fatalf("Expected offspring length 5, got %d and %d", len(seq1), len(seq2))
	}

	// Verify uniqueness
	verifyPermutation := func(seq SequenceGenome) {
		seen := make(map[int]bool)
		for _, v := range seq {
			if v < 1 || v > 5 {
				t.Errorf("Value %d is out of bounds [1, 5]", v)
			}
			if seen[v] {
				t.Errorf("Duplicate value %d found in permutation %v", v, seq)
			}
			seen[v] = true
		}
	}
	verifyPermutation(seq1)
	verifyPermutation(seq2)
}

func TestCrossovers_CompositeDelegation(t *testing.T) {
	p1 := make(CompositeGenome)
	p1["categorical"] = &CategoricalGenome[TestEnv, OperatorsTestState]{
		GeneIndices: []int{0, 1},
	}
	p1["route"] = SequenceGenome{1, 2, 3, 4}

	p2 := make(CompositeGenome)
	p2["categorical"] = &CategoricalGenome[TestEnv, OperatorsTestState]{
		GeneIndices: []int{1, 0},
	}
	p2["route"] = SequenceGenome{4, 3, 2, 1}

	// Verify Order Crossover composite delegation
	ox := OrderCrossover{}
	off1, off2 := ox.Crossover(p1, p2)

	comp1 := off1.(CompositeGenome)
	comp2 := off2.(CompositeGenome)

	// Verify sequence key is crossed over using OX (retains uniqueness)
	r1 := comp1["route"].(SequenceGenome)
	seen := make(map[int]bool)
	for _, v := range r1 {
		if seen[v] {
			t.Errorf("Duplicate %d in composite sequence offspring", v)
		}
		seen[v] = true
	}

	// Verify categorical key is crossed over using SinglePointCrossover (or fallback)
	cat1 := comp1["categorical"].(*CategoricalGenome[TestEnv, OperatorsTestState])
	cat2 := comp2["categorical"].(*CategoricalGenome[TestEnv, OperatorsTestState])
	if len(cat1.GeneIndices) != 2 || len(cat2.GeneIndices) != 2 {
		t.Errorf("Categorical indices length corrupted, expected 2")
	}
}
