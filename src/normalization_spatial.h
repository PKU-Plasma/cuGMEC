#pragma once
#include "normalization_moment.h"
#include <array>
#include <queue>

namespace particle_normalization {

struct SpatialInput {
    int nx, ny, nz;
    bool axisymmetric, realSampling;
    double dx, dy, dz, x0, y0, z0, rejectionBound, markerCount;
};
// Along a radial interpolation cell, J and N are linear. Split their product
// at the rejection bound before integrating: two Gauss rules can otherwise
// both miss a narrow unsaturated interval and falsely report convergence.
inline long double radialProposalIntegral(long double j0, long double j1, long double n0, long double n1,
                                          long double bound, bool realSampling) {
    if (!realSampling)
        return (j0 + j1) / 2;
    const long double a = (j1 - j0) * (n1 - n0);
    const long double b = j0 * (n1 - n0) + n0 * (j1 - j0);
    const long double c = j0 * n0;
    std::vector<long double> breaks = {0, 1};
    if (a == 0) {
        if (b != 0)
            addBreakpoint(breaks, (bound - c) / b);
    } else {
        const long double discriminant = b * b - 4 * a * (c - bound);
        if (discriminant > 0) {
            const long double root = -0.5L * (b + std::copysign(std::sqrt(discriminant), b));
            addBreakpoint(breaks, root / a);
            if (root != 0)
                addBreakpoint(breaks, (c - bound) / root);
        }
    }
    sortBreakpoints(breaks);
    auto product = [&](long double x) { return (a * x + b) * x + c; };
    long double integral = 0;
    for (size_t i = 1; i < breaks.size(); ++i) {
        const long double left = breaks[i - 1], right = breaks[i], mid = (left + right) / 2;
        integral +=
            (right - left) * (product(mid) >= bound ? bound : (product(left) + 4 * product(mid) + product(right)) / 6);
    }
    return integral;
}
// Each native cell has trilinear J,N,T,B. Corner index is 4*x + 2*y + z.
using FieldsAtPoint = std::array<double, 4>;
using SpatialCell = std::array<FieldsAtPoint, 8>;
using SpatialMoments = std::pair<long double, long double>;
inline FieldsAtPoint interpolateCell(const SpatialCell& cell, double x, double y, double z) {
    FieldsAtPoint value = {};
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            for (int k = 0; k < 2; ++k) {
                const double weight = (i ? x : 1 - x) * (j ? y : 1 - y) * (k ? z : 1 - z);
                for (int f = 0; f < 4; ++f)
                    value[f] += weight * cell[4 * i + 2 * j + k][f];
            }
    return value;
}
inline SpatialMoments quadraticRange(long double p0, long double p1, long double p2) {
    // p0,p1,p2 are Bernstein coefficients, not midpoint samples.
    long double low = std::min(p0, p2), high = std::max(p0, p2);
    const long double a = p0 - 2 * p1 + p2, b = 2 * (p1 - p0);
    if (a != 0) {
        const long double x = -b / (2 * a);
        if (x > 0 && x < 1) {
            const long double value = (a * x + b) * x + p0;
            low = std::min(low, value);
            high = std::max(high, value);
        }
    }
    return {low, high};
}
inline SpatialMoments productBounds(const SpatialCell& cell) {
    // The product of two trilinear fields has degree two in each coordinate.
    // Bernstein coefficients enclose its range. Maximize/minimize each angular
    // coefficient's radial quadratic exactly so angular subdivision tightens
    // the bounds even when the radial cell contains a cap transition.
    long double coefficients[3][3][3] = {};
    for (int a = 0; a < 8; ++a)
        for (int b = 0; b < 8; ++b) {
            const int x = a / 4 + b / 4, y = a / 2 % 2 + b / 2 % 2, z = a % 2 + b % 2;
            const int divisor = (x == 1 ? 2 : 1) * (y == 1 ? 2 : 1) * (z == 1 ? 2 : 1);
            coefficients[x][y][z] += (long double)cell[a][0] * cell[b][1] / divisor;
        }
    SpatialMoments bounds = {std::numeric_limits<long double>::infinity(), 0};
    for (int j = 0; j < 3; ++j)
        for (int k = 0; k < 3; ++k) {
            const auto range = quadraticRange(coefficients[0][j][k], coefficients[1][j][k], coefficients[2][j][k]);
            bounds.first = std::min(bounds.first, range.first);
            bounds.second = std::max(bounds.second, range.second);
        }
    return bounds;
}

struct CellEstimate {
    SpatialMoments value, error;
    bool unseenTransition;
};
template <class Moment>
inline CellEstimate estimateCell(const SpatialCell& cell, bool axisymmetric, bool realSampling, bool referenceSurface,
                                 double bound, Moment& moment) {
    SpatialMoments coarse = {0, 0}, fine = {0, 0};
    long double sampledMin = std::numeric_limits<long double>::infinity(), sampledMax = 0;
    auto sample = [&](double y, double z) {
        const auto left = interpolateCell(cell, 0, y, z);
        const auto right = interpolateCell(cell, 1, y, z);
        const long double volume = radialProposalIntegral(left[0], right[0], left[1], right[1], bound, realSampling);
        long double reference = 0;
        if (referenceSurface && left[1] > 0) {
            const long double proposal =
                realSampling ? std::min((long double)left[0] * left[1], (long double)bound) : (long double)left[0];
            reference = proposal / left[0] * moment(left[2], left[3]) * (realSampling ? 1 : left[1]);
        }
        const auto range = quadraticRange((long double)left[0] * left[1],
                                          ((long double)left[0] * right[1] + (long double)right[0] * left[1]) / 2,
                                          (long double)right[0] * right[1]);
        sampledMin = std::min(sampledMin, range.first);
        sampledMax = std::max(sampledMax, range.second);
        return SpatialMoments{volume, reference};
    };
    const int weights[5] = {1, 4, 2, 4, 1};
    for (int j = 0; j < 5; ++j)
        for (int k = 0; k < (axisymmetric ? 1 : 5); ++k) {
            const auto value = sample(j / 4., k / 4.);
            const long double w = weights[j] / 12.L * (axisymmetric ? 1 : weights[k] / 12.L);
            fine.first += w * value.first;
            fine.second += w * value.second;
            if (j % 2 == 0 && k % 2 == 0) {
                const long double wc = (j == 2 ? 4 : 1) / 6.L * (axisymmetric ? 1 : (k == 2 ? 4 : 1) / 6.L);
                coarse.first += wc * value.first;
                coarse.second += wc * value.second;
            }
        }
    // A cap kink can make the two nested Simpson errors cancel. Compare an
    // independent node set as well; the topology guard below covers wholly
    // unseen clipped/unclipped branches. These remain numerical error estimates.
    static const auto gauss = gaussLegendreRule(4);
    SpatialMoments independent = {0, 0};
    for (auto y : gauss)
        for (int k = 0; k < (axisymmetric ? 1 : 4); ++k) {
            const auto value = sample((y.first + 1) / 2, axisymmetric ? 0 : (gauss[k].first + 1) / 2);
            const long double w = y.second / 2 * (axisymmetric ? 1 : gauss[k].second / 2);
            independent.first += w * value.first;
            independent.second += w * value.second;
        }
    bool unseenTransition = false;
    if (realSampling) {
        const auto bounds = productBounds(cell);
        const long double roundoff =
            64 * std::numeric_limits<double>::epsilon() * std::max((long double)bound, bounds.second);
        unseenTransition = (sampledMax <= bound && bounds.second > bound + roundoff) ||
                           (sampledMin >= bound && bounds.first < bound - roundoff);
    }
    return {independent,
            {std::max(std::abs(fine.first - coarse.first) / 15, std::abs(independent.first - fine.first)),
             std::max(std::abs(fine.second - coarse.second) / 15, std::abs(independent.second - fine.second))},
            unseenTransition};
}

template <class Moment>
inline SpatialMoments integrateCell(const SpatialCell& cell, bool axisymmetric, bool realSampling,
                                    bool referenceSurface, double bound, Moment& moment) {
    struct Region {
        SpatialCell cell;
        CellEstimate estimate;
        double area;
        int depth;
        long double priority;
        bool operator<(const Region& other) const { return priority < other.priority; }
    };
    const auto first = estimateCell(cell, axisymmetric, realSampling, referenceSurface, bound, moment);
    const SpatialMoments scale = {first.value.first, first.value.second > 0 ? first.value.second : 1};
    auto relativeError = [&](const CellEstimate& e, double area) {
        return area * (e.error.first / scale.first + e.error.second / scale.second);
    };
    std::priority_queue<Region> pending;
    SpatialMoments total = {0, 0};
    long double error = 0;
    int unseen = 0;
    auto add = [&](SpatialCell c, CellEstimate e, double area, int depth) {
        const long double localError = relativeError(e, area);
        total.first += area * e.value.first;
        total.second += area * e.value.second;
        error += localError;
        unseen += e.unseenTransition;
        pending.push(
            {c, e, area, depth, e.unseenTransition ? std::numeric_limits<long double>::infinity() : localError});
    };
    add(cell, first, 1, 0);
    // Spend refinement on the largest remaining error. A relative test at every
    // tiny leaf needlessly over-resolves a cap boundary and can explode in 3D.
    while (unseen || error > 1e-9L) {
        const auto parent = pending.top();
        pending.pop();
        if (parent.depth == 32 || pending.size() > 100000)
            throw std::runtime_error("Spatial normalization quadrature did not converge");
        total.first -= parent.area * parent.estimate.value.first;
        total.second -= parent.area * parent.estimate.value.second;
        error -= relativeError(parent.estimate, parent.area);
        unseen -= parent.estimate.unseenTransition;
        for (int j = 0; j < 2; ++j)
            for (int k = 0; k < (axisymmetric ? 1 : 2); ++k) {
                SpatialCell child;
                for (int x = 0; x < 2; ++x)
                    for (int y = 0; y < 2; ++y)
                        for (int z = 0; z < 2; ++z)
                            child[4 * x + 2 * y + z] =
                                interpolateCell(parent.cell, x, (j + y) / 2., axisymmetric ? 0 : (k + z) / 2.);
                const auto estimate = estimateCell(child, axisymmetric, realSampling, referenceSurface, bound, moment);
                add(child, estimate, parent.area / (axisymmetric ? 2 : 4), parent.depth + 1);
            }
    }
    return total;
}

// Fields returns {J,N,T,B} on native interpolation nodes, including endpoints.
// Beta refers to the continuum inner surface, not a finite-cell CIC average.
// Exactly zero surface pressure cannot set this amplitude and is rejected;
// fresh loads must not silently fall back to a random boundary denominator.
template <class Fields, class Moment>
inline double referencePressure(const SpatialInput& a, Fields fields, Moment moment) {
    if (!(a.nx > 1 && a.ny > 0 && a.nz > 0 && a.dx > 0 && a.dy > 0 && a.dz > 0 && a.rejectionBound > 0 &&
          a.markerCount > 0))
        throw std::invalid_argument("Invalid native grid/proposal");
    SpatialMoments total = {0, 0};
    for (int i = 0; i < a.nx - 1; ++i)
        for (int j = 0; j < a.ny; ++j)
            for (int k = 0; k < (a.axisymmetric ? 1 : a.nz); ++k) {
                SpatialCell cell;
                for (int x = 0; x < 2; ++x)
                    for (int y = 0; y < 2; ++y)
                        for (int z = 0; z < 2; ++z) {
                            auto f = fields(a.x0 + (i + x) * a.dx, a.y0 + (j + y - .5) * a.dy,
                                            a.axisymmetric ? a.z0 : a.z0 + (k + z - .5) * a.dz);
                            if (!(f[0] > 0 && f[1] >= 0 && std::isfinite(f[0]) && std::isfinite(f[1])))
                                throw std::runtime_error("Invalid native spatial density/J");
                            cell[4 * x + 2 * y + z] = f;
                        }
                const auto value =
                    integrateCell(cell, a.axisymmetric, a.realSampling, i == 0, a.rejectionBound, moment);
                total.first += value.first;
                total.second += value.second;
            }
    // All cells have equal coordinate volume; radial dx cancels from the ratio.
    const double result = a.markerCount * total.second / (a.ny * a.nz) / total.first;
    if (!(total.first > 0 && total.second > 0 && result > 0 && std::isfinite(result)))
        throw std::runtime_error("Nonpositive or unrepresentable reference pressure");
    return result;
}
} // namespace particle_normalization
