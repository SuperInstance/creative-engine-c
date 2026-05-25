# creative-engine-c

> Dynamical systems for creative processes — Lorenz attractors, regime detection, and signal quality metrics in pure C

Part of the [SuperInstance](https://github.com/SuperInstance) music constraint theory ecosystem. Provides the low-level C implementation of the creative dynamics engine: chaotic attractors that drive generative musical processes, regime detection (fixed-point / periodic / chaotic), and signal quality metrics for evaluating creative output.

## What It Does

Creative output isn't random — it emerges from dynamical systems operating at the edge of chaos. This engine implements the Lorenz system as its core dynamical driver, integrating the attractor in real-time and classifying the system's regime. When the system is in a **chaotic regime**, it generates rich, unpredictable musical material. When it's **periodic**, patterns repeat. When at a **fixed point**, output stabilizes. The creative sweet spot is the transition between regimes.

The engine also implements **Kuramoto order parameters** for measuring synchronization between coupled oscillators (useful for coordinating multiple musical voices), **soft-snap** operations for smoothly pulling parameters toward targets, and **signal quality metrics** that evaluate whether the current dynamical state is producing viable creative output.

A Rust port lives at [creative-engine-rust](https://github.com/SuperInstance/creative-engine-rust).

## Key Features

- **Lorenz system integration** — 4th-order Runge-Kutta integration of the Lorenz attractor
- **Regime detection** — classifies dynamics as fixed-point, periodic, or chaotic
- **Soft snap** — exponential pull toward target values without discontinuities
- **Kuramoto order parameter** — measures oscillator synchronization in `[0, 1]`
- **Signal quality metrics** — evaluates creative viability of dynamical state
- **Coupling** — couples multiple dynamical systems for multi-voice coordination
- **Zero dependencies** — pure C, no external libraries needed

## Building

```bash
git clone https://github.com/SuperInstance/creative-engine-c.git
cd creative-engine-c
make                # Build static library + test binary
make test           # Build and run tests
make clean          # Clean build artifacts
```

### Prerequisites

- C99-compatible compiler (gcc, clang)
- make

## API Reference

### Lorenz Integration

```c
#include "creative_engine.h"

// Initialize Lorenz state with classic parameters
LorenzState state = {
    .x = 0.1, .y = 0.0, .z = 0.0,
    .sigma = 10.0, .rho = 28.0, .beta = 8.0/3.0
};

// Integrate one step (4th-order Runge-Kutta)
double dt = 0.01;
lorenz_step(&state, dt);

// Read current state
printf("x=%f y=%f z=%f\n", state.x, state.y, state.z);
```

### Regime Detection

```c
// Detect current regime from trajectory history
Regime regime = detect_regime(trajectory, trajectory_length);

switch (regime) {
    case REGIME_FIXED_POINT:  // System settled
    case REGIME_PERIODIC:     // Limit cycle
    case REGIME_CHAOTIC:      // Strange attractor
}
```

### Soft Snap

```c
// Smoothly pull a value toward a target
double current = 0.3;
double target = 1.0;
double strength = 0.1;  // pull rate
double snapped = soft_snap(current, target, strength);
```

### Kuramoto Order Parameter

```c
// Measure synchronization of N oscillators
double phases[] = {0.0, 1.2, 2.5, 3.8, 5.1};
double order = kuramoto_order_parameter(phases, 5);
// order ≈ 1.0 → fully synchronized
// order ≈ 0.0 → fully desynchronized
```

### Signal Quality

```c
// Evaluate creative viability
double quality = signal_quality(&state, history, history_length);
// Returns value in [0.0, 1.0]
```

### Coupling

```c
// Couple two Lorenz systems
LorenzState a = lorenz_init(0.1, 0.0, 0.0);
LorenzState b = lorenz_init(-0.1, 0.0, 0.0);
double coupling_strength = 0.05;

// One coupled step
lorenz_step_coupled(&a, &b, coupling_strength, dt);
```

## Architecture

```
creative_engine.c   # Full implementation
creative_engine.h   # Public API header
Makefile            # Build system
tests/
  test_engine.c     # Test suite
```

### Core Types

| Type | Description |
|---|---|
| `LorenzState` | 3D state vector + Lorenz parameters (σ, ρ, β) |
| `Regime` | Enum: `REGIME_FIXED_POINT`, `REGIME_PERIODIC`, `REGIME_CHAOTIC` |

## Testing

```bash
make test   # Build and run all tests
```

## Related Repos

- [**creative-engine-rust**](https://github.com/SuperInstance/creative-engine-rust) — Rust port of this engine
- [**flux-ffi**](https://github.com/SuperInstance/flux-ffi) — FFI bindings (Rust ↔ C) for shared backend
- [**superinstance-live**](https://github.com/SuperInstance/superinstance-live) — Live session controller using creative dynamics
- [**constraint-toolkit**](https://github.com/SuperInstance/constraint-toolkit) — Constraint satisfaction engine
- [**flux-genome**](https://github.com/SuperInstance/flux-genome) — Genetic evolution of musical genomes

## License

MIT
