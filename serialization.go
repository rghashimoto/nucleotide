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
	Genes []LocusGenePair `json:"genes"`
}

// EncodeGenome encodes a CategoricalGenome's gene IDs into a JSON byte slice.
func EncodeGenome[Env any, State any](g *CategoricalGenome[Env, State]) ([]byte, error) {
	data := GenomeData{
		Genes: make([]LocusGenePair, len(g.GeneIndices)),
	}

	for i, geneIdx := range g.GeneIndices {
		locus := g.Definition.Loci[i]
		if geneIdx < 0 || geneIdx >= len(locus.PossibleGenes) {
			return nil, fmt.Errorf("invalid gene index %d for locus %s", geneIdx, locus.ID)
		}
		data.Genes[i] = LocusGenePair{
			LocusID: locus.ID,
			GeneID:  locus.PossibleGenes[geneIdx].ID,
		}
	}

	return json.MarshalIndent(data, "", "  ")
}

// DecodeGenome decodes gene IDs from a JSON byte slice and maps them to indices in the provided Definition.
func DecodeGenome[Env any, State any](def *Definition[Env, State], data []byte) (*CategoricalGenome[Env, State], error) {
	var gData GenomeData
	if err := json.Unmarshal(data, &gData); err != nil {
		return nil, err
	}

	indices := make([]int, len(def.Loci))
	fileGenes := make(map[string]string)
	for _, pair := range gData.Genes {
		fileGenes[pair.LocusID] = pair.GeneID
	}

	for i, locus := range def.Loci {
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

	return &CategoricalGenome[Env, State]{
		Definition:  def,
		GeneIndices: indices,
	}, nil
}

// SaveGenome saves a CategoricalGenome's gene IDs to a JSON file.
func SaveGenome[Env any, State any](g *CategoricalGenome[Env, State], filename string) error {
	bytes, err := EncodeGenome(g)
	if err != nil {
		return err
	}
	return os.WriteFile(filename, bytes, 0644)
}

// LoadGenome loads gene IDs from a JSON file and maps them to indices in the provided Definition.
func LoadGenome[Env any, State any](def *Definition[Env, State], filename string) (*CategoricalGenome[Env, State], error) {
	bytes, err := os.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	return DecodeGenome(def, bytes)
}
