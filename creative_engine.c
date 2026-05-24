#include "creative_engine.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ── Regime Detection ────────────────────────────────────────── */

regime_t regime_from_rho(double rho) {
    if (rho < -0.01)  return REGIME_FIXED_POINT;
    if (rho <  0.01)  return REGIME_PERIODIC;
    return REGIME_CHAOTIC;
}

double optimal_epsilon(regime_t regime, double expertise) {
    /* Clamp expertise to [0,1] */
    if (expertise < 0.0) expertise = 0.0;
    if (expertise > 1.0) expertise = 1.0;

    double base;
    switch (regime) {
        case REGIME_FIXED_POINT: base = 0.1;  break;  /* mostly snap */
        case REGIME_PERIODIC:    base = 0.3;  break;
        case REGIME_CHAOTIC:     base = 0.5;  break;
        default:                 base = 0.3;  break;
    }
    /* Higher expertise → lower ε* (more precision, less exploration) */
    return base * (1.0 - 0.6 * expertise);
}

/* ── Soft Snap ───────────────────────────────────────────────── */

double soft_snap(double x, double epsilon) {
    return (1.0 - epsilon) * round(x) + epsilon * x;
}

/* ── Sigmoid ─────────────────────────────────────────────────── */

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_steep(double x, double k) {
    return 1.0 / (1.0 + exp(-k * x));
}

/* ── Kuramoto Order Parameter ────────────────────────────────── */

double kuramoto_order(const double *phases, size_t n) {
    if (n == 0) return 0.0;
    double sum_cos = 0.0, sum_sin = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum_cos += cos(phases[i]);
        sum_sin += sin(phases[i]);
    }
    double r = sqrt((sum_cos / n) * (sum_cos / n) + (sum_sin / n) * (sum_sin / n));
    return r;
}

/* ── Lorenz System (RK4) ─────────────────────────────────────── */

static lorenz_state_t lorenz_deriv(lorenz_state_t s, lorenz_params_t p) {
    return (lorenz_state_t){
        .x = p.sigma * (s.y - s.x),
        .y = s.x * (p.rho - s.z) - s.y,
        .z = s.x * s.y - p.beta * s.z
    };
}

static lorenz_state_t ls_add(lorenz_state_t a, lorenz_state_t b) {
    return (lorenz_state_t){ .x = a.x+b.x, .y = a.y+b.y, .z = a.z+b.z };
}

static lorenz_state_t ls_scale(lorenz_state_t s, double k) {
    return (lorenz_state_t){ .x = s.x*k, .y = s.y*k, .z = s.z*k };
}

lorenz_state_t lorenz_step(lorenz_state_t s, lorenz_params_t p, double dt) {
    lorenz_state_t k1 = lorenz_deriv(s, p);
    lorenz_state_t k2 = lorenz_deriv(ls_add(s, ls_scale(k1, dt/2.0)), p);
    lorenz_state_t k3 = lorenz_deriv(ls_add(s, ls_scale(k2, dt/2.0)), p);
    lorenz_state_t k4 = lorenz_deriv(ls_add(s, ls_scale(k3, dt)),      p);
    return (lorenz_state_t){
        .x = s.x + (dt/6.0)*(k1.x + 2*k2.x + 2*k3.x + k4.x),
        .y = s.y + (dt/6.0)*(k1.y + 2*k2.y + 2*k3.y + k4.y),
        .z = s.z + (dt/6.0)*(k1.z + 2*k2.z + 2*k3.z + k4.z)
    };
}

void lorenz_integrate(lorenz_state_t s0, lorenz_params_t p, double dt,
                      size_t n, lorenz_state_t *out) {
    out[0] = s0;
    for (size_t i = 1; i <= n; i++) {
        out[i] = lorenz_step(out[i-1], p, dt);
    }
}

/* ── Quality Metrics ─────────────────────────────────────────── */

double signal_novelty(const double *sig, size_t n) {
    if (n < 2) return 0.0;
    double sum = 0.0;
    for (size_t i = 1; i < n; i++) {
        sum += fabs(sig[i] - sig[i-1]);
    }
    return sum / (n - 1);
}

double signal_coherence(const double *sig, size_t n) {
    if (n < 3) return 1.0;
    /* Compute diffs, then variance of diffs */
    double sum_d = 0.0;
    for (size_t i = 1; i < n; i++) {
        sum_d += sig[i] - sig[i-1];
    }
    double mean_d = sum_d / (n - 1);

    double var_d = 0.0;
    for (size_t i = 1; i < n; i++) {
        double d = (sig[i] - sig[i-1]) - mean_d;
        var_d += d * d;
    }
    var_d /= (n - 1);

    /* Normalize: coherence = 1 / (1 + var) */
    return 1.0 / (1.0 + var_d);
}

double signal_quality(const double *sig, size_t n) {
    double q = signal_novelty(sig, n) * signal_coherence(sig, n);
    /* Normalize novelty to [0,1] range — assume typical max ~5.0 */
    double norm_novelty = signal_novelty(sig, n) / 5.0;
    if (norm_novelty > 1.0) norm_novelty = 1.0;
    q = norm_novelty * signal_coherence(sig, n);
    if (q < 0.0) q = 0.0;
    if (q > 1.0) q = 1.0;
    return q;
}

/* ── Creative Network ────────────────────────────────────────── */

creative_network_t network_create(const int *layer_sizes, int n_layers) {
    creative_network_t net;
    memset(&net, 0, sizeof(net));
    net.n_layers = n_layers < CREATIVE_NET_MAX_LAYERS ? n_layers : CREATIVE_NET_MAX_LAYERS;

    int total = 0;
    for (int l = 0; l < net.n_layers; l++) {
        net.nodes_per_layer[l] = layer_sizes[l];
        total += layer_sizes[l];
    }

    /* Set up hierarchical coupling: each node couples to nodes in adjacent layers */
    int offset = 0;
    for (int l = 0; l < net.n_layers; l++) {
        int next_offset = offset + net.nodes_per_layer[l];
        if (l + 1 < net.n_layers) {
            /* Couple each node in this layer to all nodes in next layer */
            for (int i = offset; i < next_offset; i++) {
                int next_end = next_offset + net.nodes_per_layer[l+1];
                for (int j = next_offset; j < next_end; j++) {
                    if (i < CREATIVE_NET_MAX_NODES && j < CREATIVE_NET_MAX_NODES) {
                        double w = 1.0 / net.nodes_per_layer[l+1];
                        net.coupling[i][j] = w;
                        net.coupling[j][i] = w * 0.5;  /* weaker feedback */
                    }
                }
            }
        }
        offset = next_offset;
    }
    return net;
}

void network_step(creative_network_t *net, double *activations) {
    int total = 0;
    for (int l = 0; l < net->n_layers; l++) total += net->nodes_per_layer[l];

    double *new_act = (double *)calloc(total, sizeof(double));
    if (!new_act) return;

    for (int i = 0; i < total && i < CREATIVE_NET_MAX_NODES; i++) {
        double sum = 0.0;
        double wsum = 0.0;
        for (int j = 0; j < total && j < CREATIVE_NET_MAX_NODES; j++) {
            if (net->coupling[i][j] != 0.0) {
                sum  += net->coupling[i][j] * activations[j];
                wsum += net->coupling[i][j];
            }
        }
        new_act[i] = (wsum > 0.0) ? (activations[i] + 0.1 * sum) : activations[i];
    }

    memcpy(activations, new_act, total * sizeof(double));
    free(new_act);
}

/* ── Creative Thermostat ─────────────────────────────────────── */

thermostat_t thermostat_create(double target_quality, double learning_rate) {
    return (thermostat_t){
        .epsilon       = 0.5,
        .target        = target_quality,
        .learning_rate = learning_rate,
        .min_eps       = 0.01,
        .max_eps       = 0.99
    };
}

double thermostat_update(thermostat_t *t, double measured_quality) {
    double error = t->target - measured_quality;
    /* If quality < target, increase exploration; if > target, decrease */
    t->epsilon += t->learning_rate * error;
    if (t->epsilon < t->min_eps) t->epsilon = t->min_eps;
    if (t->epsilon > t->max_eps) t->epsilon = t->max_eps;
    return t->epsilon;
}
