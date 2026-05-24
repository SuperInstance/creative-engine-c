# Creative Dynamics Engine (C)

C implementation of the Creative Dynamics Engine — a computational framework for modeling creative processes using dynamical systems theory.

## Features

- **Regime Detection**: Classify dynamics as fixed-point, periodic, or chaotic from Lyapunov exponent
- **Soft Snap**: Continuous interpolation between categorical and continuous processing
- **Sigmoid Gates**: Universal gate function with tunable steepness
- **Kuramoto Order Parameter**: Measure synchronization in coupled oscillator systems
- **Lorenz System**: RK4-integrated chaotic dynamics
- **Quality Metrics**: Novelty × coherence scoring for creative signals
- **Hierarchical Networks**: Layered coupling with feedforward and feedback connections
- **Creative Thermostat**: Adaptive exploration rate based on quality feedback

## Build

```bash
gcc -O2 -Wall -Wextra creative_engine.c test_creative.c -o test_creative -lm
./test_creative
```

## Test Results

41 tests covering all subsystems — regime detection, soft snap, sigmoid, Kuramoto, Lorenz integration, quality metrics, network coupling, and thermostat adaptation.
