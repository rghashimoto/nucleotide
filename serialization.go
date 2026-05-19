package nucleotide

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
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

// SaveGenome saves a CategoricalGenome's gene IDs to a JSON file.
func SaveGenome[E any](g *CategoricalGenome[E], filename string) error {
	data := GenomeData{
		Genes: make([]LocusGenePair, len(g.GeneIndices)),
	}

	for i, geneIdx := range g.GeneIndices {
		locus := g.Definition.Loci[i]
		if geneIdx < 0 || geneIdx >= len(locus.PossibleGenes) {
			return fmt.Errorf("invalid gene index %d for locus %s", geneIdx, locus.ID)
		}
		data.Genes[i] = LocusGenePair{
			LocusID: locus.ID,
			GeneID:  locus.PossibleGenes[geneIdx].ID,
		}
	}

	bytes, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return err
	}
	return ioutil.WriteFile(filename, bytes, 0644)
}

// LoadGenome loads gene IDs from a JSON file and maps them to indices in the provided Definition.
func LoadGenome[E any](def *Definition[E], filename string) (*CategoricalGenome[E], error) {
	bytes, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	var data GenomeData
	if err := json.Unmarshal(bytes, &data); err != nil {
		return nil, err
	}

	indices := make([]int, len(def.Loci))
	// Create a map for faster lookup if needed, but for small genomes O(N*M) is fine.
	// We'll use a map of LocusID -> GeneID from the file.
	fileGenes := make(map[string]string)
	for _, pair := range data.Genes {
		fileGenes[pair.LocusID] = pair.GeneID
	}

	for i, locus := range def.Loci {
		geneID, ok := fileGenes[locus.ID]
		if !ok {
			return nil, fmt.Errorf("locus %s not found in file", locus.ID)
		}

		// Find the index of the geneID in this locus
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

	return &CategoricalGenome[E]{
		Definition:  def,
		GeneIndices: indices,
	}, nil
}
