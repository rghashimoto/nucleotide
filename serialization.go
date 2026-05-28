package nucleotide

import (
	"encoding/json"
	"fmt"
	"os"
)

// LocusGenePair maps a Locus ID to a selected Gene ID.
type LocusGenePair struct {
	LocusID string `json:"locus_id"`
	GeneID  string `json:"gene_id"`
}

// GenomeData is the serializable format of a CategoricalGenome using IDs.
type GenomeData struct {
	Genes     []LocusGenePair           `json:"genes"`
	Sequences map[string]SequenceGenome `json:"sequences,omitempty"`
}

// EncodeGenome encodes a Genome's gene IDs into a JSON byte slice.
func EncodeGenome(g Genome) ([]byte, error) {
	var pairs []LocusGenePair
	var seqs map[string]SequenceGenome

	if comp, ok := g.(CompositeGenome); ok {
		if serializable, ok := comp["categorical"].(interface{ GetGenePairs() []LocusGenePair }); ok {
			pairs = serializable.GetGenePairs()
		}
		seqs = make(map[string]SequenceGenome)
		for k, sub := range comp {
			if seq, ok := sub.(SequenceGenome); ok {
				seqs[k] = seq
			}
		}
	} else if serializable, ok := g.(interface{ GetGenePairs() []LocusGenePair }); ok {
		pairs = serializable.GetGenePairs()
	} else {
		return nil, fmt.Errorf("unsupported genome type for encoding: %T", g)
	}

	data := GenomeData{
		Genes: pairs,
	}

	if len(seqs) > 0 {
		data.Sequences = seqs
	}

	return json.MarshalIndent(data, "", "  ")
}

// DecodeGenome decodes gene IDs from a JSON byte slice and maps them to indices in the provided Definition.
func DecodeGenome[Env any, State any](def *Definition[Env, State], data []byte) (Genome, error) {
	var gData GenomeData
	if err := json.Unmarshal(data, &gData); err != nil {
		return nil, err
	}

	indices := make([]int, len(def.Loci))
	fileGenes := make(map[string]string)
	for _, pair := range gData.Genes {
		fileGenes[pair.LocusID] = pair.GeneID
	}

	var hasSequence bool
	for _, locus := range def.Loci {
		if locus.Type == LocusSequence {
			hasSequence = true
		}
	}

	for i, locus := range def.Loci {
		if locus.Type == LocusSequence {
			continue
		}

		geneID, ok := fileGenes[locus.ID]
		if !ok {
			return nil, fmt.Errorf("locus %s not found in data", locus.ID)
		}

		found := false
		for j, gene := range locus.PossibleGenes {
			if gene.ID == geneID {
				indices[i] = j
				found = true
				break
			}
		}
		if !found {
			return nil, fmt.Errorf("gene %s not found in locus %s", geneID, locus.ID)
		}
	}

	catG := &CategoricalGenome[Env, State]{
		Definition:  def,
		GeneIndices: indices,
	}

	if hasSequence {
		comp := make(CompositeGenome)
		comp["categorical"] = catG
		for _, locus := range def.Loci {
			if locus.Type == LocusSequence {
				if gData.Sequences == nil {
					return nil, fmt.Errorf("sequence data missing")
				}
				seq, exists := gData.Sequences[locus.ID]
				if !exists {
					return nil, fmt.Errorf("sequence for locus %s missing", locus.ID)
				}
				comp[locus.ID] = seq
			}
		}
		return comp, nil
	}

	return catG, nil
}

// SaveGenome saves a Genome's gene IDs to a JSON file.
func SaveGenome(g Genome, filename string) error {
	bytes, err := EncodeGenome(g)
	if err != nil {
		return err
	}
	return os.WriteFile(filename, bytes, 0644)
}

// LoadGenome loads gene IDs from a JSON file and maps them to indices in the provided Definition.
func LoadGenome[Env any, State any](def *Definition[Env, State], filename string) (Genome, error) {
	bytes, err := os.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	return DecodeGenome(def, bytes)
}
