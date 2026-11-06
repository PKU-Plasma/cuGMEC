#pragma once
// Velocity moments for native analytic particle loading. The pressure functional
// is K = (mass * v_parallel^2 + mu * B) / 2 = mass * v^2 * (1 + pitch^2) / 4.
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace particle_normalization {

struct VelocityInput {
    int distribution; // Maxwell=0, Slowing0..3=1..4
    bool realSampling;
    double mass, temperature, B, vmin, vmax, beamSpeed, cutoffWidth;
    double lambda0, lambdaWidthSquared, energyUnit; // energyUnit = KEV/(MP*VA0^2)
};

inline std::vector<std::pair<long double, long double>> gaussLegendreRule(int n) {
    std::vector<std::pair<long double, long double>> out(n);
    const long double pi = std::acos(-1.L);
    for (int i = 0; i < (n + 1) / 2; ++i) {
        long double z = std::cos(pi * (i + .75L) / (n + .5L)), dp = 0;
        for (int iter = 0; iter < 64; ++iter) {
            long double p = 1, q = 0;
            for (int j = 1; j <= n; ++j) {
                long double r = q;
                q = p;
                p = ((2 * j - 1) * z * q - (j - 1) * r) / j;
            }
            dp = n * (z * p - q) / (z * z - 1);
            long double next = z - p / dp;
            if (std::abs(next - z) < 8 * std::numeric_limits<long double>::epsilon()) {
                z = next;
                break;
            }
            if (iter == 63)
                throw std::runtime_error("Gauss root failed");
            z = next;
        }
        long double weight = 2 / ((1 - z * z) * dp * dp);
        out[i] = {-z, weight};
        out[n - 1 - i] = {z, weight};
    }
    return out;
}
inline void addBreakpoint(std::vector<long double>& p, long double x) {
    if (std::isfinite(x) && x > p.front() && x < p.back())
        p.insert(p.end() - 1, x);
}
inline void sortBreakpoints(std::vector<long double>& p) {
    std::sort(p.begin(), p.end());
    p.erase(std::unique(p.begin(), p.end()), p.end());
}
template <class F>
inline std::pair<long double, long double> integrateMoments(const std::vector<long double>& breaks, int n, F f) {
    auto rule = gaussLegendreRule(n);
    long double norm = 0, moment = 0;
    for (size_t k = 1; k < breaks.size(); ++k) {
        long double half = (breaks[k] - breaks[k - 1]) / 2, mid = (breaks[k] + breaks[k - 1]) / 2;
        for (auto nw : rule) {
            auto v = f(mid + half * nw.first);
            norm += half * nw.second * v.first;
            moment += half * nw.second * v.second;
        }
    }
    return {norm, moment};
}
inline long double velocityMoment(const VelocityInput& a, int n) {
    const bool cutoff = a.distribution == 2 || a.distribution == 4;
    const bool pitchDependent = a.distribution == 3 || a.distribution == 4;
    if (a.distribution < 0 || a.distribution > 4 ||
        !(a.mass > 0 && a.temperature > 0 && a.B > 0 && a.vmin >= 0 && a.vmax > a.vmin && a.energyUnit > 0))
        throw std::invalid_argument("Invalid native distribution parameters");
    if (cutoff && !(a.cutoffWidth > 0))
        throw std::invalid_argument("Nonpositive cutoff width");
    if (pitchDependent && !(a.lambdaWidthSquared > 0))
        throw std::invalid_argument("Nonpositive pitch width");
    std::vector<long double> speed = {a.vmin, a.vmax}, pitch = {0, 1};
    long double scale = a.distribution == 0 ? std::sqrt(a.temperature * a.energyUnit / a.mass) : a.temperature;
    for (long double factor : {.125L, .25L, .5L, 1.L, 2.L, 4.L, 8.L, 16.L, 32.L})
        addBreakpoint(speed, factor * scale);
    if (cutoff)
        for (int k = -12; k <= 12; ++k)
            addBreakpoint(speed, a.beamSpeed + k * a.cutoffWidth);
    if (pitchDependent)
        for (int k = -12; k <= 12; ++k) {
            long double sq = 1 - a.B * (a.lambda0 + k * std::sqrt(a.lambdaWidthSquared));
            if (sq > 0)
                addBreakpoint(pitch, std::sqrt(sq));
        }
    sortBreakpoints(speed);
    sortBreakpoints(pitch);
    // Remove constant exponent offsets so very small yet valid truncated tails
    // retain relative accuracy. Restore offsets for the uniform-proposal weight.
    long double speedShift = a.distribution == 0 ? -a.vmin * a.vmin / (2 * scale * scale) : 0;
    long double pitchShift = 0;
    if (pitchDependent) {
        long double nearest = std::clamp((long double)a.lambda0, 0.L, 1.L / a.B);
        pitchShift = -(nearest - a.lambda0) * (nearest - a.lambda0) / a.lambdaWidthSquared;
    }
    auto sv = integrateMoments(speed, n, [&](long double v) {
        long double f = a.distribution == 0 ? std::exp(-v * v / (2 * scale * scale) - speedShift) /
                                                  std::pow((long double)a.temperature, 1.5L)
                                            : 1 / (v * v * v + std::pow((long double)a.temperature, 3));
        if (cutoff)
            f *= std::erfc((v - a.beamSpeed) / a.cutoffWidth);
        return std::make_pair(f * v * v, f * v * v * v * v);
    });
    auto pv = integrateMoments(pitch, n, [&](long double p) {
        long double lambda = (1 - p * p) / a.B;
        long double f = pitchDependent
                            ? std::exp(-(lambda - a.lambda0) * (lambda - a.lambda0) / a.lambdaWidthSquared - pitchShift)
                            : 1;
        return std::make_pair(f, f * (1 + p * p));
    });
    if (!(sv.first > 0 && pv.first > 0))
        throw std::runtime_error("Unresolved velocity support");
    if (a.realSampling)
        return a.mass / 4 * (sv.second / sv.first) * (pv.second / pv.first);
    long double proposal = (std::pow((long double)a.vmax, 3) - std::pow((long double)a.vmin, 3)) / 3;
    return a.mass / 4 * sv.second / proposal * pv.second * std::exp(speedShift + pitchShift);
}
inline double pressureMoment(const VelocityInput& a, double tolerance = 1e-10) {
    long double previous = velocityMoment(a, 16);
    for (int n : {32, 64, 128, 256}) {
        long double current = velocityMoment(a, n);
        if (current > 0 && std::isfinite(current) && std::abs(current - previous) <= tolerance * std::abs(current)) {
            const double result = current;
            if (!(result > 0 && std::isfinite(result)))
                throw std::runtime_error("Velocity moment is outside double precision range");
            return result;
        }
        previous = current;
    }
    throw std::runtime_error("Velocity moment quadrature did not converge");
}
} // namespace particle_normalization
