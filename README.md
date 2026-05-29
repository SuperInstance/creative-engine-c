# creative-engine-c

Dynamical systems for creative processes — Lorenz attractors, regime detection (fixed-point/periodic/chaotic), Kuramoto synchronization, signal quality metrics, and adaptive thermostats in pure C99.

## What This Gives You

- **Lorenz system** — 4th-order Runge-Kutta integration of the Lorenz attractor for chaotic creative drivers
- **Regime detection** — Classify system state as fixed-point, periodic, or chaotic
- **Kuramoto coupling** — Synchronize multiple oscillators for ensemble coherence
- **Quality metrics** — Novelty, coherence, and combined quality scores for generated material
- **Adaptive thermostat** — Automatically tune exploration rate based on output quality
- **Zero dependencies** — C99, only `<math.h>` and `<stdlib.h>`

## Quick Start

### Lorenz system

```c
#include "creative_engine.h"

lorenz_state_t state = { .x = 0.1, .y = 0.0, .z = 0.0 };
lorenz_params_t params = LORENZ_DEFAULT;  // σ=10, ρ=28, β=8/3

// Single RK4 step
state = lorenz_step(state, params, 0.01);

// Full trajectory
#define N 10000
lorenz_state_t trajectory[N + 1];
trajectory[0] = (lorenz_state_t){ .x = 1.0, .y = 1.0, .z = 1.0 };
lorenz_integrate(trajectory[0], params, 0.01, N, trajectory);
```

### Regime detection and quality

```c
// Detect which regime the system is in
regime_t regime = detect_regime(trajectory, N + 1);
// REGIME_CHAOTIC for ρ=28

// Quality metrics on output
quality_t q = compute_quality(trajectory, N + 1);
printf("novelty=%.3f coherence=%.3f quality=%.3f\n", q.novelty, q.coherence, q.quality);
```

## Build & Test

```bash
make              # build static library + test binary
make test         # build and run tests
make clean
```

## API Reference

| Function | Description |
|----------|-------------|
| `lorenz_step(state, params, dt)` | Single RK4 step (returns new state) |
| `lorenz_integrate(init, params, dt, n, out)` | Integrate n steps into trajectory array |
| `detect_regime(trajectory, n)` | Classify as `REGIME_FIXED`, `REGIME_PERIODIC`, or `REGIME_CHAOTIC` |
| `compute_quality(trajectory, n)` | Compute novelty, coherence, quality metrics |
| `kuramoto_step(phases, omegas, k, dt, n)` | One Kuramoto synchronization step |

## How It Fits

- **[creative-engine-rust](https://github.com/SuperInstance/creative-engine-rust)** — Rust port with additional coupling and builder API
- **[constraint-hamiltonian](https://github.com/SuperInstance/constraint-hamiltonian)** — Hamiltonian dynamics for constrained creative trajectories
- **[counterpoint-engine-c](https://github.com/SuperInstance/counterpoint-engine-c)** — Use Lorenz output as input for counterpoint generation
- **[flux-algebra-c](https://github.com/SuperInstance/flux-algebra-c)** — Map Lorenz coordinates to harmonic space

## Testing

45 tests covering Lorenz integration, regime detection, Kuramoto synchronization, quality metrics, and edge cases.

```bash
make test
```

## Installation

```bash
git clone https://github.com/SuperInstance/creative-engine-c.git
cd creative-engine-c
make
```

Requires C99 compiler (gcc, clang).

## License

MIT

Part of the [SuperInstance OpenConstruct](https://github.com/SuperInstance) ecosystem.
