#include <cstdint>
#include <iostream>
#include <cuda_runtime.h>  // Core CUDA runtime API
#include <device_launch_parameters.h>  // Required for kernel launch parameters
#include <cstdint>
#include <iostream>

struct Float10 {
    uint16_t data;

    static constexpr int SIGN_BITS = 1;
    static constexpr int EXP_BITS = 3;
    static constexpr int MANT_BITS = 6;
    static constexpr int EXP_BIAS = 3;

    static constexpr uint16_t SIGN_MASK = 0x200;
    static constexpr uint16_t EXP_MASK = 0x1C0;
    static constexpr uint16_t MANT_MASK = 0x03F;

    __host__ __device__ Float10() : data(0) {}

    __host__ __device__ Float10(float value) {
        uint32_t bits = *reinterpret_cast<uint32_t*>(&value);
        uint16_t sign = (bits >> 31) & 0x1;
        int32_t exponent = ((bits >> 23) & 0xFF) - 127 + EXP_BIAS;
        uint16_t mantissa = (bits >> 17) & MANT_MASK;

        if (exponent <= 0) {
            exponent = 0;
            mantissa = 0;
        }
        else if (exponent >= (1 << EXP_BITS) - 1) {
            exponent = (1 << EXP_BITS) - 1;
            mantissa = 0;
        }

        data = (sign << 9) | ((exponent & 0x7) << 6) | (mantissa & MANT_MASK);
    }

    __host__ __device__ operator float() const {
        return toFloat();
    }

    __host__ __device__ float toFloat() const {
        uint16_t sign = (data & SIGN_MASK) >> 9;
        int32_t exponent = ((data & EXP_MASK) >> 6) - EXP_BIAS + 127;
        uint32_t mantissa = (data & MANT_MASK) << 17;

        uint32_t floatBits = (sign << 31) | ((exponent & 0xFF) << 23) | mantissa;
        return *reinterpret_cast<float*>(&floatBits);
    }

    __host__ __device__ Float10 operator+(const Float10& other) const {
        return Float10(this->toFloat() + other.toFloat());
    }

    __host__ __device__ Float10 operator*(const Float10& other) const {
        return Float10(this->toFloat() * other.toFloat());
    }

    __host__ __device__ Float10 operator/(const Float10& other) const {
        return Float10(this->toFloat() / other.toFloat());
    }

    __host__ __device__ Float10 operator-(const Float10& other) const {
        return Float10(this->toFloat() - other.toFloat());
    }


    __host__ __device__ float operator+(const float& other) const {
        return (this->toFloat() + other);
    }

    __host__ __device__ float operator*(const float& other) const {
        return (this->toFloat() * other);
    }

    __host__ __device__ float operator/(const float& other) const {
        return (this->toFloat() / other);
    }

    __host__ __device__ float operator-(const float& other) const {
        return (this->toFloat() - other);
    }

    __host__ void print() const { std::cout << "Float10(" << toFloat() << ")" << std::endl; }
};
