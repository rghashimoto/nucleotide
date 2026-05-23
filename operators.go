package nucleotide

// Selector defines the interface for selecting individuals from a population.
type Selector interface {
	// We use any here because Selector might work with different Individual types.
	// However, usually we want it to be specific.
	// Since Selector is an interface, and Go doesn't support generic methods in interfaces,
	// we have a few options. One is to make Selector generic too.
	Select(pop interface{}) interface{}
}

// Crossoverer defines the interface for combining two parents into offspring.
type Crossoverer interface {
	Crossover(p1, p2 Genome) (Genome, Genome)
}

// Mutator defines the interface for introducing random changes to a genome.
type Mutator interface {
	Mutate(g Genome) Genome
}
