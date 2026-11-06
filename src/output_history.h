#pragma once
#include <array>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace output_history {
template <size_t Count>
void validateModes(const std::array<int, Count>& modes, int nz) {
    for (size_t n = 0; n < Count; ++n) {
        if (modes[n] < 0 || modes[n] > nz / 2)
            throw std::invalid_argument("Output Fourier index must be between zero and gridNz/2");
        for (size_t previous = 0; previous < n; ++previous)
            if (modes[n] == modes[previous])
                throw std::invalid_argument("Duplicate output Fourier index");
    }
}

// Same index convention as numpy.fft.rfft(history, axis=-1) / nz.
// Input [time,y,x,z]; output [time,y,x,mode,(real,imag)]. No coordinate phase shift.
template <typename Real, size_t Count>
void writeModes(const std::string& path, const std::vector<Real>& history, size_t samples,
                int ny, int nx, int nz, const std::array<int, Count>& modes) {
    validateModes(modes, nz);
    const size_t plane = static_cast<size_t>(ny) * nx;
    if (samples > history.size() / (plane * nz))
        throw std::invalid_argument("Output history is shorter than its recorded sample count");
    std::vector<std::array<double, 2>> weights(Count * nz);
    for (size_t mode = 0; mode < Count; ++mode)
        for (int k = 0; k < nz; ++k) {
            const double phase = -2 * std::acos(-1.0) * modes[mode] * k / nz;
            weights[mode * nz + k] = {std::cos(phase) / nz, std::sin(phase) / nz};
            if (modes[mode] == 0)
                weights[mode * nz + k] = {1.0 / nz, 0};
            else if (nz % 2 == 0 && modes[mode] == nz / 2)
                weights[mode * nz + k] = {(k % 2 ? -1.0 : 1.0) / nz, 0};
        }
    std::vector<Real> frame(plane * Count * 2);
    std::ofstream output;
    output.exceptions(std::ios::failbit | std::ios::badbit);
    output.open(path, std::ios::binary);
    for (size_t t = 0; t < samples; ++t) {
        for (size_t cell = 0; cell < plane; ++cell)
            for (size_t mode = 0; mode < Count; ++mode) {
                double real = 0, imag = 0;
                for (int k = 0; k < nz; ++k) {
                    const double value = history[(t * plane + cell) * nz + k];
                    real += value * weights[mode * nz + k][0];
                    imag += value * weights[mode * nz + k][1];
                }
                frame[(cell * Count + mode) * 2] = static_cast<Real>(real);
                frame[(cell * Count + mode) * 2 + 1] = static_cast<Real>(imag);
            }
        output.write(reinterpret_cast<const char*>(frame.data()), frame.size() * sizeof(Real));
    }
    output.close();
}
}
