#pragma once
#include <array>

// Output policy only; restart input remains controlled by ifContinue.
namespace output_options {
inline constexpr bool writeCheckpoints = true;
inline constexpr bool writeFinalFields = true;
// Empty: original full 3D histories. Otherwise, nonnegative FFT indices to retain.
inline constexpr std::array<int, 0> toroidalModes = {};
}
