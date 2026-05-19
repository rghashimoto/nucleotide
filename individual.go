package nucleotide

import (
	"context"
	"math/rand"
)

// Individual represents a candidate solution in the population.
type Individual[E any] struct {
	Genome  Genome
	Fitness float64
}

// NewIndividual creates a new individual with the given genome.
func NewIndividual[E any](genome Genome) *Individual[E] {
	return &Individual[E]{
		Genome: genome,
	}
}

// GetParameter returns the value of a parameter gene at a specific locus ID.
func (ind *Individual[E]) GetParameter(locusID string) interface{} {
	if cg, ok := ind.Genome.(*CategoricalGenome[E]); ok {
		for i, locus := range cg.Definition.Loci {
			if locus.ID == locusID && locus.Type == LocusParameter {
				geneIdx := cg.GeneIndices[i]
				return locus.PossibleGenes[geneIdx].Value
			}
		}
	}
	return nil
}

// Express executes the behavioral genes based on the configuration loci, with access to the environment and a cancellable context.
func (ind *Individual[E]) Express(ctx context.Context, env E) {
	cg, ok := ind.Genome.(*CategoricalGenome[E])
	if !ok {
		return
	}

	// 1. Identify behavioral loci and config loci
	behavioralLoci := []*Locus[E]{}
	behavioralIndices := []int{}
	selectedGeneIDs := []string{}
	selectedGeneIndices := []int{}

	var executionOrder interface{}

	for i, locus := range cg.Definition.Loci {
		geneIdx := cg.GeneIndices[i]
		switch locus.Type {
		case LocusBehavioral:
			behavioralLoci = append(behavioralLoci, locus)
			behavioralIndices = append(behavioralIndices, i)
			selectedGeneIDs = append(selectedGeneIDs, locus.PossibleGenes[geneIdx].ID)
			selectedGeneIndices = append(selectedGeneIndices, geneIdx)
		case LocusConfig:
			if locus.ID == "Execution Order" {
				executionOrder = locus.PossibleGenes[geneIdx].Value
			}
		}
	}

	// 2. Determine execution order
	order := make([]int, len(behavioralIndices))
	copy(order, behavioralIndices)

	seqCtx := SequencingContext[E]{
		BehavioralLoci:      behavioralLoci,
		SelectedGeneIDs:     selectedGeneIDs,
		SelectedGeneIndices: selectedGeneIndices,
	}

	switch v := executionOrder.(type) {
	case string:
		if v == "random" {
			rand.Shuffle(len(order), func(i, j int) {
				order[i], order[j] = order[j], order[i]
			})
		}
		// "sequential" is default
	case func(SequencingContext[E]) []int:
		relOrder := v(seqCtx)
		absOrder := make([]int, len(relOrder))
		for i, relIdx := range relOrder {
			absOrder[i] = behavioralIndices[relIdx]
		}
		order = absOrder
	}

	// 3. Execute callbacks
	callCtx := Context[E]{Ctx: ctx, Individual: ind, Env: env}
	for _, idx := range order {
		// Check for cancellation before executing each gene
		select {
		case <-ctx.Done():
			return
		default:
		}

		locus := cg.Definition.Loci[idx]
		geneIdx := cg.GeneIndices[idx]
		if geneIdx >= 0 && geneIdx < len(locus.PossibleGenes) {
			if callback := locus.PossibleGenes[geneIdx].Callback; callback != nil {
				callback(callCtx)
			}
		}
	}
}

// Save saves the individual's genome to a JSON file.
func (ind *Individual[E]) Save(filename string) error {
	if cg, ok := ind.Genome.(*CategoricalGenome[E]); ok {
		return SaveGenome(cg, filename)
	}
	return nil
}

// Population is a collection of individuals.
type Population[E any] []*Individual[E]

// Best returns the individual with the highest fitness.
func (p Population[E]) Best() *Individual[E] {
	if len(p) == 0 {
		return nil
	}
	best := p[0]
	for _, ind := range p[1:] {
		if ind.Fitness > best.Fitness {
			best = ind
		}
	}
	return best
}

// AverageFitness returns the average fitness.
func (p Population[E]) AverageFitness() float64 {
	if len(p) == 0 {
		return 0
	}
	var total float64
	for _, ind := range p {
		total += ind.Fitness
	}
	return total / float64(len(p))
}
