package main

import (
	"context"
	"fmt"

	"github.com/rghashimoto/nucleotide"
)

// WeatherClimate represents the weather conditions on an island.
type WeatherClimate struct {
	Name        string
	WindDrag    float64 // Extra energy cost per km due to wind resistance
	RainPenalty float64 // Speed reduction penalty due to rain safety constraints
}

// DroneState tracks actions performed during expression.
type DroneState struct {
	ActionsExecuted []string
}

func main() {
	fmt.Println("=================================================================")
	fmt.Println("      NUCLEOTIDE CONCURRENT MULTI-ISLAND CLIMATE SOLVER          ")
	fmt.Println("=================================================================")

	// 1. Define 3 heterogeneous weather environments (Option C)
	// Each island will run a separate simulator with a different climate
	climates := []WeatherClimate{
		{Name: "Island 1: Sunny & Calm", WindDrag: 0.0, RainPenalty: 0.0},
		{Name: "Island 2: High Winds", WindDrag: 1.5, RainPenalty: 0.0},
		{Name: "Island 3: Heavy Monsoon", WindDrag: 0.5, RainPenalty: 0.4},
	}

	// 2. Define the genome architecture schema
	def := nucleotide.NewDefinition[WeatherClimate, *DroneState]()

	// Parameter Loci (Battery and Cruise Speed)
	batteryLocus := def.AddLocus("BatteryCapacity", nucleotide.LocusParameter)
	batteryLocus.AddParameterGene("EcoBattery", 100.0)
	batteryLocus.AddParameterGene("ExtendedBattery", 250.0)

	speedLocus := def.AddLocus("CruiseSpeed", nucleotide.LocusParameter)
	speedLocus.AddParameterGene("EcoSpeed", 40.0)  // km/h
	speedLocus.AddParameterGene("PowerSpeed", 80.0) // km/h

	// Sequence Locus (Evolving delivery sequence of 4 major waypoints)
	routeLocus := def.AddLocus("WaypointsRoute", nucleotide.LocusSequence)
	routeLocus.AddSequenceGene("range", 1, 4)

	// 3. Define the Fitness Evaluation Function
	// The fitness is evaluated locally on each island using its specific WeatherClimate!
	fitnessFunc := func(g nucleotide.Genome, env WeatherClimate) []float64 {
		// Retrieve battery and speed parameters
		var battery float64 = 100.0
		var speed float64 = 40.0

		if comp, ok := g.(nucleotide.CompositeGenome); ok {
			if cat, ok := comp["categorical"].(interface {
				GetDefinition() interface{}
				GetIndices() []int
			}); ok {
				d := cat.GetDefinition().(*nucleotide.Definition[WeatherClimate, *DroneState])
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
			if seq, ok := comp["WaypointsRoute"].(nucleotide.SequenceGenome); ok {
				route = seq
			}
		}

		if len(route) == 0 {
			return []float64{0.0}
		}

		// Calculate total energy consumption and time spent
		// Waypoint distances: 15km, 25km, 10km, 30km
		distances := []float64{15.0, 25.0, 10.0, 30.0}
		totalTime := 0.0
		deliveredWaypoints := 0.0

		// Weather climate parameters
		effectiveSpeed := speed * (1.0 - env.RainPenalty)
		energyPerKm := 1.0 + env.WindDrag
		if speed > 60.0 {
			energyPerKm += 1.5 // fast speed consumes more base battery
		}

		for _, waypointID := range route {
			wIdx := waypointID - 1
			dist := distances[wIdx]

			travelTime := dist / effectiveSpeed
			totalTime += travelTime

			energyRequired := dist * energyPerKm
			if battery < energyRequired {
				break // Battery depleted
			}
			battery -= energyRequired
			deliveredWaypoints += 1.0
		}

		// High delivery value is prioritized, penalized slightly by travel duration
		return []float64{deliveredWaypoints*100.0 - (totalTime * 5.0)}
	}

	// 4. Setup Multi-Island Configuration (Option C: Distributed Climate Factory)
	config := nucleotide.EngineConfig[WeatherClimate, *DroneState]{
		PopulationSize: 20,
		MaxGenerations: 40,
		FitnessFunc:    fitnessFunc,
		Selector:       nucleotide.GenericTournamentSelector[WeatherClimate, *DroneState]{Size: 3},
		Elitism:        1,
	}

	// Factory building Option C heterogeneous climates per island
	envFactory := func(islandIndex int) WeatherClimate {
		return climates[islandIndex]
	}

	miConfig := nucleotide.MultiIslandEngineConfig[WeatherClimate, *DroneState]{
		NumIslands:        3,
		MigrationInterval: 5, // Perform migration epoch every 5 generations
		MigrationRate:     2, // Move top 2 individuals
		MigrationTopology: nucleotide.TopologyRing,
		MigrationPolicy:   nucleotide.PolicyBestReplaceWorst,
		EngineConfig:      config,
		EnvFactory:        envFactory,
	}

	miEngine, err := nucleotide.NewMultiIslandEngine(miConfig)
	if err != nil {
		fmt.Printf("Error building MultiIslandEngine: %v\n", err)
		return
	}

	// Print island environment specialization configurations
	fmt.Println("Configured Sub-Populations (Islands):")
	for i, island := range miEngine.Islands {
		fmt.Printf(" - Island %d Environment: %s (Wind Drag = %.1f, Rain Penalty = %.1f)\n",
			i+1, island.Config.Env.Name, island.Config.Env.WindDrag, island.Config.Env.RainPenalty)
	}

	// 5. Run the evolution optimizer concurrently
	fmt.Println("\nRunning concurrent multi-island evolution with migrations...")
	bestInd, err := miEngine.Run(def)
	if err != nil {
		fmt.Printf("Evolution failed: %v\n", err)
		return
	}

	fmt.Printf("\nMulti-Island Evolution Complete!\n")
	fmt.Printf("Global Best Evolved Fitness: %.2f\n", bestInd.Fitness[0])

	// Express behaviors
	state := &DroneState{ActionsExecuted: make([]string, 0)}
	bestInd.State = state
	bestInd.Express(context.Background(), climates[0]) // Express in ideal sunny climate

	bestRoute := bestInd.GetSequence("WaypointsRoute")
	fmt.Printf("Evolved Optimal Waypoint Visited Route: %v\n", bestRoute)

	bestBattery := bestInd.GetParameter("BatteryCapacity")
	bestSpeed := bestInd.GetParameter("CruiseSpeed")
	fmt.Printf("Evolved Generalist Configuration: Battery = %v Wh, Cruise Speed = %v km/h\n", bestBattery, bestSpeed)
	fmt.Println("=================================================================")
}
