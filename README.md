# creative-engine-c

Dynamical systems for creative processes — Lorenz attractors, regime detection, Kuramoto synchronization, and signal quality metrics in pure C99.

Part of the [SuperInstance](https://github.com/SuperInstance) ecosystem. The Lorenz system acts as a chaotic driver for generative musical processes. Regime detection (fixed-point / periodic / chaotic) determines the character of creative output. Signal quality metrics evaluate whether the current state produces viable material. A creative thermostat adaptively tunes exploration rate.

Zero dependencies — only `<math.h>` and `<stdlib.h>`.

## Building

```bash
make              # build static library + test binary
make test         # build and run tests
make clean
```

Requires a C99 compiler (gcc, clang) and make.

## Lorenz System

4th-order Runge-Kutta integration of the Lorenz attractor:

```
dx/dt = σ(y − x)
dy/dt = x(ρ − z) − y
dz/dt = xy − βz
```

```c
#include "creative_engine.h"

// State and parameters are separate structs
lorenz_state_t state = { .x = 0.1, .y = 0.0, .z = 0.0 };
lorenz_params_t params = LORENZ_DEFAULT;  // σ=10, ρ=28, β=8/3

// Single RK4 step
state = lorenz_step(state, params, 0.01);
printf("x=%.4f y=%.4f z=%.4f\n", state.x, state.y, state.z);

// Integrate a full trajectory
#define N 10000
lorenz_state_t trajectory[N + 1];
trajectory[0] = (lorenz_state_t){ .x = 1.0, .y = 1.0, .z = 1.0 };
lorenz_integrate(trajectory[0], params, 0.01, N, trajectory);
// trajectory[i] contains state at t = i * 0.01
```

`lorenz_step` returns the new state (value, not pointer) — the original is not modified.

## Regime Detection

Classifies dynamics from the largest Lyapunov exponent ρ:

| ρ | Regime | Creative behavior |
|---|---|---|
| ρ < −0.01 | `REGIME_FIXED_POINT` | Output stabilizes |
| −0.01 ≤ ρ ≤ 0.01 | `REGIME_PERIODIC` | Repeating patterns |
| ρ > 0.01 | `REGIME_CHAOTIC` | Rich, unpredictable material |

```c
double lyapunov_exp = compute_lyapunov(trajectory, N);  // your computation
regime_t regime = regime_from_rho(lyapunov_exp);
```

## Soft Snap

Continuous snap function that interpolates between rounding and identity:

```
C(x, ε) = (1 − ε) · round(x) + ε · x
```

- ε → 0: hard snap to nearest integer
- ε → 1: pass-through (no snapping)
- ε between: smooth blend

```c
double x = 2.7;
soft_snap(x, 0.0);  // → 3.0  (hard round)
soft_snap(x, 0.5);  // → 2.85
soft_snap(x, 1.0);  // → 2.7  (identity)
```

## Optimal Epsilon

Computes the ideal exploration rate for a given regime and expertise level (0 = novice, 1 = expert):

```c
double eps = optimal_epsilon(REGIME_CHAOTIC, 0.5);
// Base rates: fixed-point=0.1, periodic=0.3, chaotic=0.5
// Higher expertise → lower ε (more precision, less exploration)
```

## Sigmoid

```c
double y = sigmoid(2.0);              // 1 / (1 + e⁻ˣ)
double y = sigmoid_steep(2.0, 5.0);   // 1 / (1 + e⁻ᵏˣ), k controls steepness
```

## Kuramoto Order Parameter

Measures synchronization of N coupled oscillators from their phases:

```
r = |1/N · Σ exp(i·θⱼ)|
```

- r ≈ 1.0: fully synchronized
- r ≈ 0.0: fully desynchronized

```c
double phases[] = { 0.0, 0.1, 0.05, 6.2, 0.08 };
double r = kuramoto_order(phases, 5);
// r ≈ 0.85 → mostly synchronized

double spread[] = { 0.0, 1.57, 3.14, 4.71, 6.28 };
double r2 = kuramoto_order(spread, 5);
// r2 ≈ 0.0 → uniformly spread
```

## Signal Quality Metrics

Three metrics for evaluating creative output quality:

```c
double signal[] = { 1.0, 1.5, 2.1, 2.8, 3.3, 3.9, 4.5 };

// Novelty: mean absolute difference of consecutive values
double nov = signal_novelty(signal, 7);
// Higher → more variation between samples

// Coherence: 1 / (1 + var(differences))
double coh = signal_coherence(signal, 7);
// Higher → smoother evolution (low variance in step sizes)

// Quality: normalized_novelty × coherence, clamped to [0, 1]
double q = signal_quality(signal, 7);
// Balances novelty and coherence — high quality needs both
```

Quality = 0 means either no variation (stuck) or chaotic noise. Quality ≈ 1 means smooth, evolving output.

## Creative Network

Hierarchical network of coupled nodes. Each layer connects to the next with weighted coupling, and weaker feedback flows backward.

```c
int layers[] = { 4, 8, 4, 2 };  // 4 layers: 4→8→4→2 nodes
creative_network_t net = network_create(layers, 4);

// Initialize activations for all 18 nodes
int total = 4 + 8 + 4 + 2;
double activations[18] = { /* initial values */ };

// One coupling step: each node updates based on weighted neighbors
network_step(&net, activations);
// activations[] is modified in-place
```

Coupling weights: forward = `1/n_next_layer`, backward = `0.5 × forward`.

## Creative Thermostat

Adapts exploration rate (epsilon) based on measured quality vs. target:

```c
thermostat_t t = thermostat_create(
    0.7,    // target quality
    0.05    // learning rate
);

// Each frame/step:
double quality = signal_quality(sig, sig_len);
double eps = thermostat_update(&t, quality);
// quality < target → increase ε (more exploration)
// quality > target → decrease ε (more exploitation)
// ε clamped to [min_eps=0.01, max_eps=0.99]
```

## Types

```c
typedef struct {
    double x, y, z;
} lorenz_state_t;

typedef struct {
    double sigma, rho, beta;
} lorenz_params_t;

typedef enum {
    REGIME_FIXED_POINT = 0,
    REGIME_PERIODIC    = 1,
    REGIME_CHAOTIC     = 2,
    REGIME_UNKNOWN     = 3
} regime_t;

typedef struct {
    int    n_layers;
    int    nodes_per_layer[8];         // max 8 layers
    double coupling[64][64];           // max 64 nodes total
} creative_network_t;

typedef struct {
    double epsilon;        // current exploration rate
    double target;         // target quality
    double learning_rate;  // adaptation speed
    double min_eps, max_eps;
} thermostat_t;
```

## API Summary

| Function | Signature |
|---|---|
| `lorenz_step` | `lorenz_state_t lorenz_step(lorenz_state_t s, lorenz_params_t p, double dt)` |
| `lorenz_integrate` | `void lorenz_integrate(lorenz_state_t s0, lorenz_params_t p, double dt, size_t n, lorenz_state_t *out)` |
| `regime_from_rho` | `regime_t regime_from_rho(double rho)` |
| `optimal_epsilon` | `double optimal_epsilon(regime_t regime, double expertise)` |
| `soft_snap` | `double soft_snap(double x, double epsilon)` |
| `sigmoid` | `double sigmoid(double x)` |
| `sigmoid_steep` | `double sigmoid_steep(double x, double k)` |
| `kuramoto_order` | `double kuramoto_order(const double *phases, size_t n)` |
| `signal_novelty` | `double signal_novelty(const double *sig, size_t n)` |
| `signal_coherence` | `double signal_coherence(const double *sig, size_t n)` |
| `signal_quality` | `double signal_quality(const double *sig, size_t n)` |
| `network_create` | `creative_network_t network_create(const int *layer_sizes, int n_layers)` |
| `network_step` | `void network_step(creative_network_t *net, double *activations)` |
| `thermostat_create` | `thermostat_t thermostat_create(double target_quality, double learning_rate)` |
| `thermostat_update` | `double thermostat_update(thermostat_t *t, double measured_quality)` |

## Related

- [creative-engine-rust](https://github.com/SuperInstance/creative-engine-rust) — Rust port
- [superinstance-live](https://github.com/SuperInstance/superinstance-live) — Live session controller
- [flux-genome](https://github.com/SuperInstance/flux-genome) — Genetic evolution of musical genomes

## License

MIT
