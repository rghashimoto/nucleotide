package main

import (
	"context"
	"fmt"
	"math"

	"github.com/rghashimoto/nucleotide"
)

// LogisticsEnv represents the customer delivery locations and parameters.
type LogisticsEnv struct {
	CustomerDistances []float64 // Distances from depot to customer sites (1 to 5)
	CustomerValues    []float64 // Value/Priority of each customer package
	MaxAllowedTime    float64   // Maximum time before delivery window expiration
}

// LogisticsState tracks actions performed during express.
type LogisticsState struct {
	ActionsExecuted []string
	TotalTime       float64
	DeliveredValue  float64
}

func main() {
	fmt.Println("=================================================================")
	fmt.Println("         NUCLEOTIDE COMPREHENSIVE GA LOGISTICS SOLVER            ")
	fmt.Println("=================================================================")

	// 1. Setup the environment data
	// 5 customers with their distance (km) and cargo priorities
	env := LogisticsEnv{
		CustomerDistances: []float64{10.0, 25.0, 15.0, 30.0, 5.0},
		CustomerValues:    []float64{100.0, 250.0, 150.0, 400.0, 80.0},
		MaxAllowedTime:    4.0, // 4 hours limit
	}

	// 2. Define the genome architecture schema
	def := nucleotide.NewDefinition[LogisticsEnv, *LogisticsState]()

	// Type 1: LocusConfig (Framework config - Execution Order config locus is added by NewDefinition)
	// We configure custom behavioral callbacks using Behavioral Loci

	// Type 2: LocusBehavioral (Action callbacks)
	actionLocus := def.AddLocus("StartupAction", nucleotide.LocusBehavioral)
	actionLocus.AddGene("LoadCargo", func(ctx nucleotide.Context[LogisticsEnv, *LogisticsState]) {
		ctx.Individual.State.ActionsExecuted = append(ctx.Individual.State.ActionsExecuted, "Loaded cargo into Drone")
	})
	actionLocus.AddGene("QuickInspect", func(ctx nucleotide.Context[LogisticsEnv, *LogisticsState]) {
		ctx.Individual.State.ActionsExecuted = append(ctx.Individual.State.ActionsExecuted, "Performed quick pre-flight safety check")
	})

	// Type 3: LocusParameter (Drone physical configurations)
	capacityLocus := def.AddLocus("BatteryCapacity", nucleotide.LocusParameter)
	capacityLocus.AddParameterGene("StandardBattery", 100.0) // 100 Wh
	capacityLocus.AddParameterGene("ExtendedBattery", 200.0) // 200 Wh

	speedLocus := def.AddLocus("CruiseSpeed", nucleotide.LocusParameter)
	speedLocus.AddParameterGene("SlowEco", 40.0)  // 40 km/h
	speedLocus.AddParameterGene("FastPower", 80.0) // 80 km/h

	// Type 4: LocusSequence (Sequence/Permutation representing delivery route ordering)
	// We want to optimize the visiting sequence of the 5 customer locations (indexes 1 to 5)
	routeLocus := def.AddLocus("DeliveryRoute", nucleotide.LocusSequence)
	routeLocus.AddSequenceGene("range", 1, 5)

	// 3. Define the Fitness Evaluation Function
	fitnessFunc := func(g nucleotide.Genome, env LogisticsEnv) []float64 {
		// Retrieve parameters
		var battery float64 = 100.0
		var speed float64 = 40.0

		// Since g is a CompositeGenome, let's extract the parameters
		if comp, ok := g.(nucleotide.CompositeGenome); ok {
			// Find parameters inside the categorical chromosome
			if cat, ok := comp["categorical"].(interface {
				GetDefinition() interface{}
				GetIndices() []int
			}); ok {
				d := cat.GetDefinition().(*nucleotide.Definition[LogisticsEnv, *LogisticsState])
				indices := cat.GetIndices()
				for i, locus := range d.Loci {
					if locus.ID == "BatteryCapacity" {
						battery = locus.PossibleGenes[indices[i]].Value.(float64)
					}
					if locus.ID == "CruiseSpeed" {
						speed = locus.PossibleGenes[indices[i]].Value.(float64)
					}
				}
			}
		}

		// Retrieve delivery route sequence
		var route nucleotide.SequenceGenome
		if comp, ok := g.(nucleotide.CompositeGenome); ok {
			if seq, ok := comp["DeliveryRoute"].(nucleotide.SequenceGenome); ok {
				route = seq
			}
		}

		if len(route) == 0 {
			return []float64{0.0}
		}

		// Calculate total travel time and priority delivery value
		currentLocation := 0.0 // Depot is at index 0
		deliveredValue := 0.0
		timeSpent := 0.0

		// Drone physics: eco vs fast power usage
		energyPerKm := 1.0
		if speed > 60.0 {
			energyPerKm = 2.5
		}

		for _, customerID := range route {
			custIdx := customerID - 1
			distToCust := env.CustomerDistances[custIdx]

			distFromPrevious := math.Abs(distToCust - currentLocation)
			if currentLocation == 0.0 {
				distFromPrevious = distToCust
			}

			travelTime := distFromPrevious / speed
			timeSpent += travelTime

			// Check battery constraints
			energyRequired := distFromPrevious * energyPerKm
			if battery < energyRequired {
				break // Battery depleted
			}
			battery -= energyRequired

			// Check time limit
			if timeSpent > env.MaxAllowedTime {
				break // Time expired
			}

			deliveredValue += env.CustomerValues[custIdx]
			currentLocation = distToCust
		}

		return []float64{deliveredValue - (timeSpent * 5.0)}
	}

	// 4. Set up the genetic algorithm engine
	config := nucleotide.EngineConfig[LogisticsEnv, *LogisticsState]{
		PopulationSize: 50,
		MaxGenerations: 60,
		FitnessFunc:    fitnessFunc,
		Selector:       nucleotide.GenericTournamentSelector[LogisticsEnv, *LogisticsState]{Size: 4},
		Elitism:        2,
		Env:            env,
		Verbose:        false,
	}

	engine, err := nucleotide.NewEngine[LogisticsEnv, *LogisticsState](config)
	if err != nil {
		fmt.Printf("Error creating engine: %v\n", err)
		return
	}

	// 5. Run the evolution optimizer
	fmt.Println("Running evolutionary optimization...")
	bestInd, err := engine.Run(def)
	if err != nil {
		fmt.Printf("Evolution failed: %v\n", err)
		return
	}

	fmt.Printf("\nEvolution complete!\n")
	fmt.Printf("Best Route Fitness: %.2f\n", bestInd.Fitness[0])

	// Initialize individual state for expression
	state := &LogisticsState{
		ActionsExecuted: make([]string, 0),
		TotalTime:       0.0,
		DeliveredValue:  0.0,
	}
	bestInd.State = state

	// Express behaviors
	bestInd.Express(context.Background(), env)
	fmt.Println("Startup Actions Executed:")
	for _, act := range state.ActionsExecuted {
		fmt.Printf(" - %s\n", act)
	}

	// Retrieve best route sequence
	bestRoute := bestInd.GetSequence("DeliveryRoute")
	fmt.Printf("Optimal Visiting Route Order: %v\n", bestRoute)

	// Retrieve best parameters
	bestBattery := bestInd.GetParameter("BatteryCapacity")
	bestSpeed := bestInd.GetParameter("CruiseSpeed")
	fmt.Printf("Chosen Configuration: Battery = %v Wh, Cruise Speed = %v km/h\n", bestBattery, bestSpeed)

	// 6. Demonstrate Robust Multi-Chromosomal Serialization (Unified Save & Load)
	filename := "best_dispatch_plan.json"

	fmt.Printf("\nSerializing and saving best dispatch plan to %s...\n", filename)
	err = nucleotide.SaveGenome(bestInd.Genome, filename)
	if err != nil {
		fmt.Printf("Failed to save genome: %v\n", err)
		return
	}

	fmt.Println("Genome successfully saved to disk.")

	// Load the genome back using Definition to reconstruct the chromosome mappings
	fmt.Println("Deserializing and reloading the saved dispatch plan...")
	loadedGenome, err := nucleotide.LoadGenome(def, filename)
	if err != nil {
		fmt.Printf("Failed to load genome: %v\n", err)
		return
	}

	// Reconstruct a temporary loaded individual to express and verify
	loadedInd := nucleotide.NewIndividual[LogisticsEnv, *LogisticsState](loadedGenome)
	loadedRoute := loadedInd.GetSequence("DeliveryRoute")
	loadedBattery := loadedInd.GetParameter("BatteryCapacity")
	loadedSpeed := loadedInd.GetParameter("CruiseSpeed")

	fmt.Printf("\nSuccessfully Reloaded Verification:\n")
	fmt.Printf(" - Loaded Delivery Route: %v\n", loadedRoute)
	fmt.Printf(" - Loaded Battery Capacity: %v\n", loadedBattery)
	fmt.Printf(" - Loaded Cruise Speed: %v\n", loadedSpeed)
	fmt.Println("=================================================================")
}
