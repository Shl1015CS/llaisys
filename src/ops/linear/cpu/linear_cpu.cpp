#include "linear_cpu.hpp"

#include "../../../utils.hpp"

// 矩阵乘法: Y = X * W^T + bias
// X: (batch_size, in_features)
// W: (out_features, in_features)  
// Y: (batch_size, out_features)
// bias: (out_features) 可选
template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias,
             size_t batch_size, size_t in_features, size_t out_features) {
    
    // 对每个批次的每个输出特征计算
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t o = 0; o < out_features; o++) {
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                // 对半精度类型，使用float进行累加以避免精度损失
                float sum_float = 0.0f;
                
                // 计算点积: X[b,:] · W[o,:]
                for (size_t i = 0; i < in_features; i++) {
                    float x_float = llaisys::utils::cast<float>(in[b * in_features + i]);
                    float w_float = llaisys::utils::cast<float>(weight[o * in_features + i]);
                    sum_float += x_float * w_float;
                }
                
                // 添加偏置（如果有）
                if (bias != nullptr) {
                    sum_float += llaisys::utils::cast<float>(bias[o]);
                }
                
                out[b * out_features + o] = llaisys::utils::cast<T>(sum_float);
            } else {
                // 对全精度类型，直接计算
                T sum = T(0);
                
                // 计算点积: X[b,:] · W[o,:]
                for (size_t i = 0; i < in_features; i++) {
                    sum += in[b * in_features + i] * weight[o * in_features + i];
                }
                
                // 添加偏置（如果有）
                if (bias != nullptr) {
                    sum += bias[o];
                }
                
                out[b * out_features + o] = sum;
            }
        }
    }
}

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t batch_size, size_t in_features, size_t out_features) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const float *>(weight),
            bias ? reinterpret_cast<const float *>(bias) : nullptr,
            batch_size, in_features, out_features
        );
    case LLAISYS_DTYPE_BF16:
        return linear_(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(in),
            reinterpret_cast<const llaisys::bf16_t *>(weight),
            bias ? reinterpret_cast<const llaisys::bf16_t *>(bias) : nullptr,
            batch_size, in_features, out_features
        );
    case LLAISYS_DTYPE_F16:
        return linear_(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(in),
            reinterpret_cast<const llaisys::fp16_t *>(weight),
            bias ? reinterpret_cast<const llaisys::fp16_t *>(bias) : nullptr,
            batch_size, in_features, out_features
        );
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
