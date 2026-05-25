# creative-engine-c

> Dynamical systems engine for creative processes — in C.

`creative-engine-c` implements the mathematical primitives underlying the SuperInstance creative AI: chaotic attractors for generating novel patterns, synchronization metrics for measuring coherence, hierarchical networks for multi-scale creative coupling, and an adaptive thermostat that balances exploration vs. exploitation.

## Features

- **Lorenz attractor** — RK4 integration of the classic chaotic system for generating trajectories with tunable chaos
- **Regime detection** — Classify dynamics as fixed-point, periodic, or chaotic from the largest Lyapunov exponent (ρ)
- **Soft snap** — Continuous interpolation between quantization and free values: `C(x, ε) = (1−ε)·round(x) + ε·x`
- **Kuramoto order parameter** — Measure synchronization of coupled oscillators
- **Signal quality metrics** — Novelty, coherence, and their product as a unified quality score
- **Hierarchical creative network** — Multi-layer coupled dynamical system with forward and feedback connections
- **Creative thermostat** — Adaptive control of exploration rate (ε) based on quality feedback

## Building

```bash
# Build the test suite
gcc -O2 -o test_creative test_creative.c creative_engine.c -lm

# Run tests
./test_creative
```

No external dependencies beyond the C standard library and `libm`.

## Architecture

### Dynamical Regimes

The engine classifies creative dynamics into three regimes based on the largest Lyapunov exponent (ρ):

| Regime | ρ | Behavior | Optimal ε |
|---|---|---|---|
| Fixed-point | ρ < −0.01 | Converges to equilibrium | Low (0.1) — mostly snap |
| Periodic | −0.01 ≤ ρ ≤ 0.01 | Oscillates regularly | Medium (0.3) |
| Chaotic | ρ > 0.01 | Sensitive to initial conditions | High (0.5) — more exploration |

Higher expertise scales ε down: experts need less exploration noise.

### Lorenz System

The Lorenz attractor (σ=10, ρ=28, β=8/3) generates chaotic trajectories. The engine provides:

- `lorenz_step()` — single RK4 step
- `lorenz_integrate()` — full trajectory integration with configurable step size

At default parameters, the system is chaotic — generating high-novelty signals. Lowering ρ below ~24.74 produces periodic behavior, and below ~1 produces fixed-point convergence.

### Soft Snap

The core creative interpolation primitive:

```
C(x, ε) = (1 − ε) · round(x) + ε · x
```

- `ε = 0`: Full quantization (snap to nearest integer)
- `ε = 1`: No quantization (free values)
- `ε ∈ (0, 1)`: Continuous blend

This maps directly to musical creativity: ε controls how strictly a generated value follows discrete scales (quantized) vs. continuous exploration.

### Kuramoto Order Parameter

Measures synchronization among N coupled oscillators:

```
r = (1/N) |Σ e^(iθⱼ)|
```

- `r = 1`: Perfect sync (all phases identical)
- `r ≈ 0`: No sync (phases uniformly distributed)

### Signal Quality

```
quality = normalized_novelty × coherence
```

Where:
- **Novelty** = mean |x[i] − x[i−1]| — how much the signal changes
- **Coherence** = 1 / (1 + var(diffs)) — how smoothly it changes

High quality requires both novelty and coherence — the signal should change but not erratically.

### Creative Network

A hierarchical coupled network with up to 8 layers and 64 nodes:

```
Layer 0 (input) → Layer 1 → ... → Layer N (output)
                    ↑________↓  (feedback coupling)
```

Each step, activations propagate forward (weight 1/next_layer_size) and backward (weight 0.5×forward). This implements multi-scale creative dynamics where lower layers generate raw material and higher layers refine it.

### Creative Thermostat

An adaptive controller that tunes exploration rate (ε):

```c
ε ← ε + α · (target_quality − measured_quality)
```

When quality is below target, ε increases (more exploration). When above, ε decreases (more refinement). Clamped to `[min_eps, max_eps]`.

## Usage Examples

### Generate a chaotic trajectory

```c
#include "creative_engine.h"

lorenz_state_t s0 = {1.0, 1.0, 1.0};
lorenz_state_t trajectory[1001];
lorenz_integrate(s0, LORENZ_DEFAULT, 0.01, 1000, trajectory);

// Extract x-coordinate signal
double xs[1001];
for (int i = 0; i <= 1000; i++) xs[i] = trajectory[i].x;

// Measure quality
double q = signal_quality(xs, 1001);
```

### Regime-aware soft snap

```c
// Detect regime
regime_t r = regime_from_rho(0.5);  // chaotic
double eps = optimal_epsilon(r, 0.3);  // intermediate expertise

// Snap a value
double raw = 2.7;
double snapped = soft_snap(raw, eps);
// Result blends between 3.0 (quantized) and 2.7 (raw)
```

### Measure synchronization

```c
double phases[4] = {0.0, 0.1, 0.05, 0.08};  // nearly synced
double r = kuramoto_order(phases, 4);
// r ≈ 0.99 — highly synchronized
```

### Build and run a creative network

```c
int layers[] = {3, 5, 4, 2};
creative_network_t net = network_create(layers, 4);

double activations[14] = {0};
activations[0] = 1.0;  // seed input

for (int step = 0; step < 100; step++) {
    network_step(&net, activations);
}
// activations now contain the evolved creative state
```

### Adaptive exploration with thermostat

```c
thermostat_t t = thermostat_create(0.6, 0.1);  // target quality 0.6

for (int i = 0; i < 50; i++) {
    // Generate something and measure quality
    double q = measure_quality_somehow();
    
    // Adapt exploration rate
    double eps = thermostat_update(&t, q);
    
    // Use eps for soft_snap or other creative operations
}
```

## API Reference

### Regime Detection

| Function | Description |
|---|---|
| `regime_from_rho(rho)` | Classify dynamics from Lyapunov exponent |
| `optimal_epsilon(regime, expertise)` | Compute ε* for given regime and skill level |

### Soft Snap & Sigmoid

| Function | Description |
|---|---|
| `soft_snap(x, epsilon)` | `(1−ε)·round(x) + ε·x` |
| `sigmoid(x)` | Standard sigmoid `1/(1+e⁻ˣ)` |
| `sigmoid_steep(x, k)` | Sigmoid with steepness parameter |

### Synchronization

| Function | Description |
|---|---|
| `kuramoto_order(phases, n)` | Kuramoto order parameter r ∈ [0, 1] |

### Lorenz System

| Function | Description |
|---|---|
| `lorenz_step(s, p, dt)` | Single RK4 step |
| `lorenz_integrate(s0, p, dt, n, out)` | Full trajectory (n+1 states) |

### Quality Metrics

| Function | Description |
|---|---|
| `signal_novelty(sig, n)` | Mean absolute difference of consecutive values |
| `signal_coherence(sig, n)` | Inverse of diff variance |
| `signal_quality(sig, n)` | `novelty × coherence` ∈ [0, 1] |

### Creative Network

| Function | Description |
|---|---|
| `network_create(layer_sizes, n_layers)` | Build hierarchical coupled network |
| `network_step(net, activations)` | Propagate activations one step |

### Thermostat

| Function | Description |
|---|---|
| `thermostat_create(target, lr)` | Initialize adaptive controller |
| `thermostat_update(t, quality)` | Adapt ε based on measured quality |

## Related Repos

- **[creative-engine-rust](https://github.com/SuperInstance/creative-engine-rust)** — Rust port of this engine
- **[constraint-toolkit](https://github.com/SuperInstance/constraint-toolkit)** — Dial space definitions and constraint solving
- **[superinstance-live](https://github.com/SuperInstance/superinstance-live)** — Live session controller using the creative engine
- **[flux-genome](https://github.com/SuperInstance/flux-genome)** — Genetic algorithm framework for evolving traditions

## License

MIT
