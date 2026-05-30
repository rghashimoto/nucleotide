package nucleotide

import (
	"math"
)

// AdaptiveMutationController defines the interface for dynamically scaling or adjusting mutation rates.
type AdaptiveMutationController[Env any, State any] interface {
	// GetMutationScaler returns a scaling factor for the mutation rate.
	GetMutationScaler(e *Engine[Env, State]) float64
}

// SigmoidDiversityFeedbackController implements a smooth logistic (sigmoid) feedback loop
// centered on a target diversity.
type SigmoidDiversityFeedbackController[Env any, State any] struct {
	TargetDiversity float64 // The desired diversity level (e.g., 0.3)
	Sensitivity     float64 // Controls the steepness of the curve (e.g., 10.0)
	MinScaler       float64 // Minimum allowed scaling factor (e.g., 0.1)
	MaxScaler       float64 // Maximum allowed scaling factor (e.g., 5.0)
}

// NewSigmoidDiversityFeedbackController creates a new SigmoidDiversityFeedbackController.
func NewSigmoidDiversityFeedbackController[Env any, State any](target, sensitivity, minS, maxS float64) *SigmoidDiversityFeedbackController[Env, State] {
	if sensitivity == 0 {
		sensitivity = 10.0
	}
	if minS == 0 {
		minS = 0.1
	}
	if maxS == 0 {
		maxS = 5.0
	}
	return &SigmoidDiversityFeedbackController[Env, State]{
		TargetDiversity: target,
		Sensitivity:     sensitivity,
		MinScaler:       minS,
		MaxScaler:       maxS,
	}
}

// GetMutationScaler computes the scaling factor using a sigmoid curve.
func (c *SigmoidDiversityFeedbackController[Env, State]) GetMutationScaler(e *Engine[Env, State]) float64 {
	div := e.genotypicDiversity()
	exponent := c.Sensitivity * (div - c.TargetDiversity)
	scaler := c.MinScaler + (c.MaxScaler-c.MinScaler)/(1.0+math.Exp(exponent))
	return scaler
}

// MutationScheduleType defines the supported temporal schedules.
type MutationScheduleType int

const (
	// ScheduleExponentialDecay continuously decays the mutation rate.
	ScheduleExponentialDecay MutationScheduleType = iota
	// ScheduleCosineAnnealing cycles the mutation rate using cosine curves.
	ScheduleCosineAnnealing
)

// TemporalScheduleController scales the mutation rate based on generation count.
type TemporalScheduleController[Env any, State any] struct {
	Type        MutationScheduleType
	InitialRate float64 // Initial scaling factor (typically 1.0 or higher)
	FinalRate   float64 // Minimum baseline scaling factor (e.g., 0.1)
	CycleLength int     // Generation period for Cosine Annealing (e.g., 20 generations)
}

// GetMutationScaler calculates the scheduling-based scaling factor.
func (c *TemporalScheduleController[Env, State]) GetMutationScaler(e *Engine[Env, State]) float64 {
	maxGens := e.Config.MaxGenerations
	if maxGens <= 0 {
		maxGens = 100 // fallback
	}
	currentGen := e.Generation

	switch c.Type {
	case ScheduleExponentialDecay:
		if currentGen >= maxGens {
			return c.FinalRate
		}
		initial := c.InitialRate
		if initial <= 0 {
			initial = 1.0
		}
		final := c.FinalRate
		if final <= 0 {
			final = 0.1
		}
		ratio := final / initial
		progress := float64(currentGen) / float64(maxGens)
		return initial * math.Pow(ratio, progress)

	case ScheduleCosineAnnealing:
		cycleLen := c.CycleLength
		if cycleLen <= 0 {
			cycleLen = 10
		}
		initial := c.InitialRate
		if initial <= 0 {
			initial = 1.0
		}
		final := c.FinalRate
		if final <= 0 {
			final = 0.1
		}
		denom := cycleLen
		if cycleLen > 1 {
			denom = cycleLen - 1
		}
		cycleGen := currentGen % cycleLen
		cosProgress := math.Cos(math.Pi * float64(cycleGen) / float64(denom))
		return final + 0.5*(initial-final)*(1.0+cosProgress)

	default:
		return 1.0
	}
}

// RechenbergController adjusts mutation scale based on the ratio of successful mutations.
type RechenbergController[Env any, State any] struct {
	Interval           int     // Number of generations between adjustments (e.g., 5)
	TargetSuccessRatio float64 // Targeted ratio of successful mutations (default 0.2)
	IncreaseFactor     float64 // Multiplier to increase mutation (default 1.22)
	DecreaseFactor     float64 // Multiplier to decrease mutation (default 0.82)
	MinScaler          float64 // Minimum allowed scaling factor (default 0.1)
	MaxScaler          float64 // Maximum allowed scaling factor (default 5.0)
	currentScaler      float64
}

// NewRechenbergController creates a new RechenbergController.
func NewRechenbergController[Env any, State any](interval int, targetSuccessRatio float64) *RechenbergController[Env, State] {
	if interval <= 0 {
		interval = 5
	}
	if targetSuccessRatio <= 0 {
		targetSuccessRatio = 0.2
	}
	return &RechenbergController[Env, State]{
		Interval:           interval,
		TargetSuccessRatio: targetSuccessRatio,
		IncreaseFactor:     1.22,
		DecreaseFactor:     0.82,
		MinScaler:          0.1,
		MaxScaler:          5.0,
		currentScaler:      1.0,
	}
}

// GetMutationScaler checks the success ratio over the interval and scales.
func (c *RechenbergController[Env, State]) GetMutationScaler(e *Engine[Env, State]) float64 {
	if c.currentScaler <= 0 {
		c.currentScaler = 1.0
	}

	// Update only at interval boundaries when there has been active evaluation
	if e.Generation > 0 && e.Generation%c.Interval == 0 && e.TotalMutations > 0 {
		ratio := float64(e.SuccessfulMutations) / float64(e.TotalMutations)

		if ratio > c.TargetSuccessRatio {
			c.currentScaler *= c.IncreaseFactor
		} else if ratio < c.TargetSuccessRatio {
			c.currentScaler *= c.DecreaseFactor
		}

		// Clamp the scaling factors
		if c.currentScaler < c.MinScaler {
			c.currentScaler = c.MinScaler
		}
		if c.currentScaler > c.MaxScaler {
			c.currentScaler = c.MaxScaler
		}

		// Reset success counters for the next interval
		e.SuccessfulMutations = 0
		e.TotalMutations = 0
	}

	return c.currentScaler
}

// SelfAdaptiveController implements individual-level mutation rate adaptation.
type SelfAdaptiveController[Env any, State any] struct {
	LearningRate float64 // Learning rate tau parameter (default 0.15)
	MinRate      float64 // Minimum allowed mutation probability (default 0.005)
	MaxRate      float64 // Maximum allowed mutation probability (default 0.3)
}

// NewSelfAdaptiveController creates a new SelfAdaptiveController.
func NewSelfAdaptiveController[Env any, State any](learningRate, minRate, maxRate float64) *SelfAdaptiveController[Env, State] {
	if learningRate <= 0 {
		learningRate = 0.15
	}
	if minRate <= 0 {
		minRate = 0.005
	}
	if maxRate <= 0 {
		maxRate = 0.3
	}
	return &SelfAdaptiveController[Env, State]{
		LearningRate: learningRate,
		MinRate:      minRate,
		MaxRate:      maxRate,
	}
}

// GetMutationScaler returns the baseline scale. Self-adaptation is processed per individual.
func (c *SelfAdaptiveController[Env, State]) GetMutationScaler(e *Engine[Env, State]) float64 {
	return 1.0
}
