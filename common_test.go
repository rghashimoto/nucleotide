package nucleotide

type TestEnv struct{}

type CounterState struct {
	Count int
}

type MockCrossoverer struct {
	id int
}

func (m MockCrossoverer) Crossover(p1, p2 Genome) (Genome, Genome) {
	return p1.Copy(), p2.Copy()
}

type MockMutator struct {
	id int
}

func (m MockMutator) Mutate(g Genome) Genome {
	return g.Copy()
}

type CustomSelfAdaptiveState struct {
	PreferredK int
}

func (c CustomSelfAdaptiveState) GetSelectionPreferences() (int, bool) {
	return c.PreferredK, true
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && stringContains(s, substr)
}

func stringContains(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
