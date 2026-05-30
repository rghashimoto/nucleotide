package nucleotide

import (
	"context"
	"math/rand"
	"os"
)

// Individual represents a candidate solution in the population.
type Individual[Env any, State any] struct {
	Genome           Genome
	Fitness          []float64
	State            State
	Age              int
	Rank             int
	CrowdingDistance float64
	MutationRate     float64   // Individual-specific mutation rate (self-adaptation)
	ParentFitness    []float64 // Fitness slice of the best parent for tracking success
}

// NewIndividual creates a new individual with the given genome.
func NewIndividual[Env any, State any](genome Genome) *Individual[Env, State] {
	return &Individual[Env, State]{
		Genome: genome,
	}
}

func (ind *Individual[Env, State]) getCategoricalGenome() *CategoricalGenome[Env, State] {
	if cg, ok := ind.Genome.(*CategoricalGenome[Env, State]); ok {
		return cg
	}
	if comp, ok := ind.Genome.(CompositeGenome); ok {
		if cat, ok := comp["categorical"].(*CategoricalGenome[Env, State]); ok {
			return cat
		}
	}
	return nil
}

// GetSequence returns the SequenceGenome at a specific locus ID.
func (ind *Individual[Env, State]) GetSequence(locusID string) SequenceGenome {
	if comp, ok := ind.Genome.(CompositeGenome); ok {
		if seq, ok := comp[locusID].(SequenceGenome); ok {
			return seq
		}
	}
	if seq, ok := ind.Genome.(SequenceGenome); ok {
		return seq
	}
	return nil
}

// GetParameter returns the value of a parameter gene at a specific locus ID.
func (ind *Individual[Env, State]) GetParameter(locusID string) interface{} {
	if cg := ind.getCategoricalGenome(); cg != nil {
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
func (ind *Individual[Env, State]) Express(ctx context.Context, env Env) {
	cg := ind.getCategoricalGenome()
	if cg == nil {
		return
	}

	// 1. Identify behavioral loci and config loci
	behavioralLoci := []*Locus[Env, State]{}
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

	seqCtx := SequencingContext[Env, State]{
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
	case func(SequencingContext[Env, State]) []int:
		relOrder := v(seqCtx)
		absOrder := make([]int, len(relOrder))
		for i, relIdx := range relOrder {
			absOrder[i] = behavioralIndices[relIdx]
		}
		order = absOrder
	}

	// 3. Execute callbacks
	callCtx := Context[Env, State]{Ctx: ctx, Individual: ind, Env: env}
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

// ToJSON encodes the individual's genome to a JSON byte slice.
func (ind *Individual[Env, State]) ToJSON() ([]byte, error) {
	return EncodeGenome(ind.Genome)
}

// Save saves the individual's genome to a JSON file.
func (ind *Individual[Env, State]) Save(filename string) error {
	bytes, err := ind.ToJSON()
	if err != nil {
		return err
	}
	return os.WriteFile(filename, bytes, 0644)
}

// Population is a collection of individuals.
type Population[Env any, State any] []*Individual[Env, State]

// Best returns the individual with the highest fitness.
func (p Population[Env, State]) Best() *Individual[Env, State] {
	if len(p) == 0 {
		return nil
	}

	// If ranks have been calculated (multi-objective mode), find Rank 0 with highest CrowdingDistance
	hasRanks := false
	for _, ind := range p {
		if ind.Rank != 0 {
			hasRanks = true
			break
		}
	}

	if hasRanks {
		var best *Individual[Env, State]
		for _, ind := range p {
			if ind.Rank == 0 {
				if best == nil || ind.CrowdingDistance > best.CrowdingDistance {
					best = ind
				}
			}
		}
		if best != nil {
			return best
		}
	}

	// Fallback to highest Fitness[0]
	best := p[0]
	for _, ind := range p[1:] {
		var indFit, bestFit float64
		if len(ind.Fitness) > 0 {
			indFit = ind.Fitness[0]
		}
		if len(best.Fitness) > 0 {
			bestFit = best.Fitness[0]
		}
		if indFit > bestFit {
			best = ind
		}
	}
	return best
}

// AverageFitness returns the average fitness for each objective.
func (p Population[Env, State]) AverageFitness() []float64 {
	if len(p) == 0 {
		return nil
	}

	numObjectives := len(p[0].Fitness)
	if numObjectives == 0 {
		return nil
	}

	averages := make([]float64, numObjectives)
	for _, ind := range p {
		for i := 0; i < numObjectives; i++ {
			if i < len(ind.Fitness) {
				averages[i] += ind.Fitness[i]
			}
		}
	}

	for i := range averages {
		averages[i] /= float64(len(p))
	}

	return averages
}
