#include "creative_engine.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static int g_pass = 0;
static int g_fail = 0;

#define ASSERT(cond, msg) do { \
    if (cond) { g_pass++; } \
    else { g_fail++; printf("  FAIL: %s (line %d)\n", msg, __LINE__); } \
} while(0)

#define ASSERT_APPROX(a, b, tol, msg) ASSERT(fabs((a)-(b)) < (tol), msg)

/* ────────────────────────────────────────────────────────────── */

static void test_regime_detection(void) {
    printf("Test: regime_from_rho\n");
    ASSERT(regime_from_rho(-1.0)  == REGIME_FIXED_POINT, "negative rho → fixed");
    ASSERT(regime_from_rho(0.0)   == REGIME_PERIODIC,    "zero rho → periodic");
    ASSERT(regime_from_rho(0.5)   == REGIME_CHAOTIC,     "positive rho → chaotic");
    ASSERT(regime_from_rho(-0.02) == REGIME_FIXED_POINT,  "small neg → fixed");
    ASSERT(regime_from_rho(0.001) == REGIME_PERIODIC,     "tiny pos → periodic");
}

static void test_optimal_epsilon_ordering(void) {
    printf("Test: optimal_epsilon ordering by expertise\n");
    double e_beginner = optimal_epsilon(REGIME_CHAOTIC, 0.0);
    double e_inter    = optimal_epsilon(REGIME_CHAOTIC, 0.5);
    double e_expert   = optimal_epsilon(REGIME_CHAOTIC, 1.0);
    ASSERT(e_beginner > e_inter,   "beginner ε > intermediate ε");
    ASSERT(e_inter    > e_expert,  "intermediate ε > expert ε");
    printf("    beginner=%.3f inter=%.3f expert=%.3f\n", e_beginner, e_inter, e_expert);
}

static void test_soft_snap_zero(void) {
    printf("Test: soft_snap ε=0 (full snap)\n");
    /* ε=0 → pure round */
    ASSERT_APPROX(soft_snap(2.7, 0.0), 3.0, 1e-12, "2.7 snaps to 3.0");
    ASSERT_APPROX(soft_snap(2.3, 0.0), 2.0, 1e-12, "2.3 snaps to 2.0");
    ASSERT_APPROX(soft_snap(-0.6, 0.0), -1.0, 1e-12, "-0.6 snaps to -1.0");
}

static void test_soft_snap_one(void) {
    printf("Test: soft_snap ε=1 (no snap)\n");
    /* ε=1 → identity */
    ASSERT_APPROX(soft_snap(2.7, 1.0), 2.7, 1e-12, "ε=1 → identity for 2.7");
    ASSERT_APPROX(soft_snap(-3.14, 1.0), -3.14, 1e-12, "ε=1 → identity for -3.14");
}

static void test_soft_snap_half(void) {
    printf("Test: soft_snap ε=0.5 (halfway)\n");
    double val = soft_snap(2.7, 0.5);
    double expected = 0.5 * 3.0 + 0.5 * 2.7;  /* 2.85 */
    ASSERT_APPROX(val, expected, 1e-12, "ε=0.5 → halfway between snap and raw");
}

static void test_sigmoid_midpoint(void) {
    printf("Test: sigmoid midpoint\n");
    ASSERT_APPROX(sigmoid(0.0), 0.5, 1e-12, "sigmoid(0) = 0.5");
}

static void test_sigmoid_extremes(void) {
    printf("Test: sigmoid extremes\n");
    ASSERT(sigmoid(-10.0) < 0.01,   "sigmoid(-10) ≈ 0");
    ASSERT(sigmoid(10.0)  > 0.99,   "sigmoid(10) ≈ 1");
    ASSERT(sigmoid(-100.0) < 1e-10, "sigmoid(-100) ≈ 0");
    ASSERT(sigmoid(100.0)  > 1.0-1e-10, "sigmoid(100) ≈ 1");
}

static void test_kuramoto_synced(void) {
    printf("Test: Kuramoto order — synced phases\n");
    double phases[5] = {0.1, 0.1, 0.1, 0.1, 0.1};
    double r = kuramoto_order(phases, 5);
    ASSERT_APPROX(r, 1.0, 1e-10, "identical phases → r ≈ 1");
}

static void test_kuramoto_random(void) {
    printf("Test: Kuramoto order — spread phases\n");
    double phases[4] = {0.0, M_PI/2, M_PI, 3*M_PI/2};
    double r = kuramoto_order(phases, 4);
    ASSERT(r < 0.1, "uniform spread → r ≈ 0");
}

static void test_lorenz_step(void) {
    printf("Test: Lorenz step changes state\n");
    lorenz_state_t s0 = {1.0, 1.0, 1.0};
    lorenz_state_t s1 = lorenz_step(s0, LORENZ_DEFAULT, 0.01);
    ASSERT(s1.x != s0.x || s1.y != s0.y || s1.z != s0.z, "state changes after step");
    /* Check it's not NaN */
    ASSERT(!isnan(s1.x) && !isnan(s1.y) && !isnan(s1.z), "no NaN in output");
}

static void test_chaotic_diversity(void) {
    printf("Test: chaotic regime has high diversity\n");
    lorenz_state_t trajectory[1001];
    lorenz_state_t s0 = {1.0, 1.0, 1.0};
    lorenz_integrate(s0, LORENZ_DEFAULT, 0.01, 1000, trajectory);

    double xs[1001];
    for (int i = 0; i <= 1000; i++) xs[i] = trajectory[i].x;

    double nov = signal_novelty(xs, 1001);
    printf("    chaotic x-novelty = %.4f\n", nov);
    ASSERT(nov > 0.05, "chaotic signal has significant novelty");
}

static void test_fixed_point_diversity(void) {
    printf("Test: fixed-point regime has low diversity\n");
    /* Use rho=0.5 (below critical) → fixed point attractor */
    lorenz_params_t p = {10.0, 0.5, 8.0/3.0};
    lorenz_state_t s0 = {1.0, 1.0, 1.0};

    lorenz_state_t trajectory[1001];
    lorenz_integrate(s0, p, 0.01, 1000, trajectory);

    double xs[1001];
    for (int i = 0; i <= 1000; i++) xs[i] = trajectory[i].x;

    /* Use last 100 points (settled) */
    double nov = signal_novelty(xs + 901, 100);
    printf("    fixed-point x-novelty = %.6f\n", nov);
    ASSERT(nov < 0.01, "fixed-point signal has near-zero novelty");
}

static void test_quality_constant_vs_varying(void) {
    printf("Test: quality constant vs varying signal\n");
    double constant[10] = {5,5,5,5,5,5,5,5,5,5};
    double varying[10]  = {1,3,2,4,1,5,2,3,4,2};

    double q_const = signal_quality(constant, 10);
    double q_var   = signal_quality(varying, 10);
    printf("    constant q=%.4f, varying q=%.4f\n", q_const, q_var);
    ASSERT(q_var > q_const, "varying signal has higher quality than constant");
}

static void test_network_creation(void) {
    printf("Test: network creation and coupling\n");
    int layers[] = {3, 4, 2};
    creative_network_t net = network_create(layers, 3);

    ASSERT(net.n_layers == 3, "3 layers created");
    ASSERT(net.nodes_per_layer[0] == 3, "layer 0 has 3 nodes");
    ASSERT(net.nodes_per_layer[1] == 4, "layer 1 has 4 nodes");
    ASSERT(net.nodes_per_layer[2] == 2, "layer 2 has 2 nodes");

    /* Check coupling: node 0 (layer 0) → node 3 (layer 1) should exist */
    ASSERT(net.coupling[0][3] > 0.0, "forward coupling exists");
    ASSERT(net.coupling[3][0] > 0.0, "backward coupling exists");

    /* No coupling to self */
    ASSERT(net.coupling[0][0] == 0.0, "no self-coupling");
}

static void test_network_step(void) {
    printf("Test: network step changes activations\n");
    int layers[] = {2, 2};
    creative_network_t net = network_create(layers, 2);

    double activations[4] = {1.0, 0.0, 0.0, 0.0};
    double orig[4];
    memcpy(orig, activations, sizeof(activations));

    network_step(&net, activations);

    int changed = 0;
    for (int i = 0; i < 4; i++) {
        if (fabs(activations[i] - orig[i]) > 1e-10) changed = 1;
    }
    ASSERT(changed, "activations change after network step");
}

static void test_thermostat_adaptation(void) {
    printf("Test: thermostat adapts ε over time\n");
    thermostat_t t = thermostat_create(0.5, 0.1);
    double eps0 = t.epsilon;

    /* Quality below target → ε should increase */
    thermostat_update(&t, 0.1);
    ASSERT(t.epsilon > eps0, "ε increases when quality < target");

    /* Quality above target → ε should decrease */
    double eps_high = t.epsilon;
    thermostat_update(&t, 0.9);
    ASSERT(t.epsilon < eps_high, "ε decreases when quality > target");
}

static void test_regime_epsilon_decreases(void) {
    printf("Test: ε* decreases with expertise across all regimes\n");
    for (int r = 0; r < 3; r++) {
        double e0 = optimal_epsilon(r, 0.0);
        double e1 = optimal_epsilon(r, 1.0);
        ASSERT(e0 > e1, "ε* decreases with expertise");
    }
}

static void test_kuramoto_empty(void) {
    printf("Test: Kuramoto order — empty input\n");
    ASSERT_APPROX(kuramoto_order(NULL, 0), 0.0, 1e-12, "empty → 0");
}

static void test_lorenz_integrate_length(void) {
    printf("Test: Lorenz integrate stores n+1 states\n");
    lorenz_state_t traj[101];
    lorenz_state_t s0 = {1.0, 1.0, 1.0};
    lorenz_integrate(s0, LORENZ_DEFAULT, 0.01, 100, traj);

    /* First should be s0 */
    ASSERT_APPROX(traj[0].x, 1.0, 1e-10, "traj[0] = initial state");
    /* Last should differ from first */
    ASSERT(fabs(traj[100].x - traj[0].x) > 0.01, "traj[100] ≠ traj[0]");
}

/* ────────────────────────────────────────────────────────────── */

int main(void) {
    printf("=== Creative Dynamics Engine — Test Suite ===\n\n");

    test_regime_detection();
    test_optimal_epsilon_ordering();
    test_soft_snap_zero();
    test_soft_snap_one();
    test_soft_snap_half();
    test_sigmoid_midpoint();
    test_sigmoid_extremes();
    test_kuramoto_synced();
    test_kuramoto_random();
    test_kuramoto_empty();
    test_lorenz_step();
    test_lorenz_integrate_length();
    test_chaotic_diversity();
    test_fixed_point_diversity();
    test_quality_constant_vs_varying();
    test_network_creation();
    test_network_step();
    test_thermostat_adaptation();
    test_regime_epsilon_decreases();

    printf("\n=== Results: %d passed, %d failed, %d total ===\n",
           g_pass, g_fail, g_pass + g_fail);

    return g_fail > 0 ? 1 : 0;
}
