#ifndef CREATIVE_ENGINE_H
#define CREATIVE_ENGINE_H

#include <stddef.h>
#include <stdint.h>

/* ── Regime Types ────────────────────────────────────────────── */
typedef enum {
    REGIME_FIXED_POINT = 0,
    REGIME_PERIODIC    = 1,
    REGIME_CHAOTIC     = 2,
    REGIME_UNKNOWN     = 3
} regime_t;

/* ── Core Primitives ─────────────────────────────────────────── */

/* Detect regime from largest Lyapunov exponent (ρ) */
regime_t regime_from_rho(double rho);

/* Optimal epsilon for a given regime and expertise level (0=novice..1=expert) */
double optimal_epsilon(regime_t regime, double expertise);

/* Soft snap: C(x,ε) = (1-ε)*round(x) + ε*x */
double soft_snap(double x, double epsilon);

/* Sigmoid universal gate */
double sigmoid(double x);
double sigmoid_steep(double x, double k);  /* with steepness k */

/* ── Kuramoto Order Parameter ────────────────────────────────── */
double kuramoto_order(const double *phases, size_t n);

/* ── Lorenz System ───────────────────────────────────────────── */
typedef struct {
    double x, y, z;
} lorenz_state_t;

typedef struct {
    double sigma, rho, beta;
} lorenz_params_t;

#define LORENZ_DEFAULT (lorenz_params_t){ .sigma = 10.0, .rho = 28.0, .beta = 8.0/3.0 }

/* Single RK4 step with step size dt */
lorenz_state_t lorenz_step(lorenz_state_t s, lorenz_params_t p, double dt);

/* Integrate n steps, store trajectory in out (must have n+1 slots) */
void lorenz_integrate(lorenz_state_t s0, lorenz_params_t p, double dt,
                      size_t n, lorenz_state_t *out);

/* ── Quality Metrics ─────────────────────────────────────────── */

/* Compute novelty of a signal (mean absolute difference of consecutive values) */
double signal_novelty(const double *sig, size_t n);

/* Compute coherence of a signal (1.0 - normalized variance of differences) */
double signal_coherence(const double *sig, size_t n);

/* Overall quality = novelty × coherence, clamped [0,1] */
double signal_quality(const double *sig, size_t n);

/* ── Creative Network ────────────────────────────────────────── */
#define CREATIVE_NET_MAX_LAYERS 8
#define CREATIVE_NET_MAX_NODES  64

typedef struct {
    int   n_layers;
    int   nodes_per_layer[CREATIVE_NET_MAX_LAYERS];
    double coupling[CREATIVE_NET_MAX_NODES][CREATIVE_NET_MAX_NODES];
} creative_network_t;

/* Create a hierarchical network with given layer sizes */
creative_network_t network_create(const int *layer_sizes, int n_layers);

/* Apply one coupling step: each node influenced by weighted sum of connected nodes */
void network_step(creative_network_t *net, double *activations);

/* ── Creative Thermostat ─────────────────────────────────────── */
typedef struct {
    double epsilon;       /* current exploration rate */
    double target;        /* target quality */
    double learning_rate; /* adaptation speed */
    double min_eps, max_eps;
} thermostat_t;

thermostat_t thermostat_create(double target_quality, double learning_rate);

/* Adapt epsilon based on measured quality */
double thermostat_update(thermostat_t *t, double measured_quality);

#endif /* CREATIVE_ENGINE_H */
