#include "cuSZp_kernels_1D_f32_block_size_N.h"

static __device__ inline int quantization_block_size_N(float data, float recipPrecision)
{
    int result;
    asm("{\n\t"
        ".reg .f32 dataRecip;\n\t"
        ".reg .f32 temp1;\n\t"
        ".reg .s32 s;\n\t"
        ".reg .pred p;\n\t"
        "mul.f32 dataRecip, %1, %2;\n\t"
        "setp.ge.f32 p, dataRecip, -0.5;\n\t"
        "selp.s32 s, 0, 1, p;\n\t"
        "add.f32 temp1, dataRecip, 0.5;\n\t"
        "cvt.rzi.s32.f32 %0, temp1;\n\t"
        "sub.s32 %0, %0, s;\n\t"
        "}": "=r"(result) : "f"(data), "f"(recipPrecision));
    return result;
}

static __device__ inline int get_bit_num_block_size_N(unsigned int x)
{
    int leading_zeros;
    asm("clz.b32 %0, %1;" : "=r"(leading_zeros) : "r"(x));
    return 32 - leading_zeros;
}

template <int DBLOCK>
__global__ void cuSZp_compress_kernel_1D_plain_f32_block_size_N(
    const float* const __restrict__ oriData, unsigned char* const __restrict__ cmpData,
    volatile unsigned int* const __restrict__ cmpOffset,
    volatile unsigned int* const __restrict__ locOffset,
    volatile int* const __restrict__ flag, const float eb, const size_t nbEle)
{
    __shared__ unsigned int excl_sum;
    __shared__ unsigned int base_idx;

    constexpr int SIGN_WORDS = DBLOCK / 32;
    constexpr int SIGN_BYTES = DBLOCK / 8;
    constexpr int block_num = thread_chunk / DBLOCK;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int idx = bid * blockDim.x + tid;
    const int lane = idx & 0x1f;
    const int warp = idx >> 5;
    const int rate_ofs =
        (nbEle + (size_t)tblock_size * thread_chunk - 1) /
        ((size_t)tblock_size * thread_chunk) * ((size_t)tblock_size * thread_chunk) / DBLOCK;
    const float recipPrecision = 0.5f / eb;

    if (!tid) {
        excl_sum = 0;
        base_idx = 0;
    }
    __syncthreads();

    int base_start_idx;
    int base_block_start_idx, base_block_end_idx;
    int quant_chunk_idx;
    int block_idx;
    int currQuant, lorenQuant, prevQuant, maxQuant;
    int absQuant[thread_chunk];
    unsigned int sign_flag[block_num * SIGN_WORDS];
    int fixed_rate[block_num];
    unsigned int thread_ofs = 0;
    float4 tmp_buffer;
    uchar4 tmp_char;

    base_start_idx = warp * thread_chunk * 32;
    for (int j = 0; j < block_num; j++) {
        base_block_start_idx = base_start_idx + j * (32 * DBLOCK) + lane * DBLOCK;
        base_block_end_idx = base_block_start_idx + DBLOCK;
        for (int sw = 0; sw < SIGN_WORDS; sw++) sign_flag[j * SIGN_WORDS + sw] = 0;
        block_idx = base_block_start_idx / DBLOCK;
        prevQuant = 0;
        maxQuant = 0;

        if (base_block_end_idx < nbEle) {
            for (int i = base_block_start_idx; i < base_block_end_idx; i += 4) {
                tmp_buffer = reinterpret_cast<const float4*>(oriData)[i / 4];
                quant_chunk_idx = j * DBLOCK + (i - base_block_start_idx);

                int local_i = i - base_block_start_idx;

                currQuant = quantization_block_size_N(tmp_buffer.x, recipPrecision);
                lorenQuant = currQuant - prevQuant;
                prevQuant = currQuant;
                sign_flag[j * SIGN_WORDS + local_i / 32] |=
                    (lorenQuant < 0) << (31 - (local_i % 32));
                absQuant[quant_chunk_idx] = abs(lorenQuant);
                maxQuant = maxQuant > absQuant[quant_chunk_idx] ? maxQuant
                                                                 : absQuant[quant_chunk_idx];

                currQuant = quantization_block_size_N(tmp_buffer.y, recipPrecision);
                lorenQuant = currQuant - prevQuant;
                prevQuant = currQuant;
                sign_flag[j * SIGN_WORDS + (local_i + 1) / 32] |=
                    (lorenQuant < 0) << (31 - ((local_i + 1) % 32));
                absQuant[quant_chunk_idx + 1] = abs(lorenQuant);
                maxQuant = maxQuant > absQuant[quant_chunk_idx + 1]
                               ? maxQuant
                               : absQuant[quant_chunk_idx + 1];

                currQuant = quantization_block_size_N(tmp_buffer.z, recipPrecision);
                lorenQuant = currQuant - prevQuant;
                prevQuant = currQuant;
                sign_flag[j * SIGN_WORDS + (local_i + 2) / 32] |=
                    (lorenQuant < 0) << (31 - ((local_i + 2) % 32));
                absQuant[quant_chunk_idx + 2] = abs(lorenQuant);
                maxQuant = maxQuant > absQuant[quant_chunk_idx + 2]
                               ? maxQuant
                               : absQuant[quant_chunk_idx + 2];

                currQuant = quantization_block_size_N(tmp_buffer.w, recipPrecision);
                lorenQuant = currQuant - prevQuant;
                prevQuant = currQuant;
                sign_flag[j * SIGN_WORDS + (local_i + 3) / 32] |=
                    (lorenQuant < 0) << (31 - ((local_i + 3) % 32));
                absQuant[quant_chunk_idx + 3] = abs(lorenQuant);
                maxQuant = maxQuant > absQuant[quant_chunk_idx + 3]
                               ? maxQuant
                               : absQuant[quant_chunk_idx + 3];
            }
        } else {
            if (base_block_start_idx >= nbEle) {
                quant_chunk_idx = j * DBLOCK;
                for (int i = 0; i < DBLOCK; i++) absQuant[quant_chunk_idx + i] = 0;
            } else {
                int remainbEle = nbEle - base_block_start_idx;

                for (int i = 0; i < remainbEle; i++) {
                    quant_chunk_idx = j * DBLOCK + i;
                    currQuant =
                        quantization_block_size_N(oriData[base_block_start_idx + i], recipPrecision);
                    lorenQuant = currQuant - prevQuant;
                    prevQuant = currQuant;
                    sign_flag[j * SIGN_WORDS + i / 32] |= (lorenQuant < 0) << (31 - (i % 32));
                    absQuant[quant_chunk_idx] = abs(lorenQuant);
                    maxQuant = maxQuant > absQuant[quant_chunk_idx] ? maxQuant
                                                                     : absQuant[quant_chunk_idx];
                }

                for (int i = remainbEle; i < DBLOCK; i++) absQuant[j * DBLOCK + i] = 0;
            }
        }

        fixed_rate[j] = get_bit_num_block_size_N(maxQuant);
        thread_ofs += (fixed_rate[j]) ? (SIGN_BYTES + fixed_rate[j] * SIGN_BYTES) : 0;
        cmpData[block_idx] = (unsigned char)fixed_rate[j];
        __syncthreads();
    }

    #pragma unroll 5
    for (int i = 1; i < 32; i <<= 1) {
        int tmp = __shfl_up_sync(0xffffffff, thread_ofs, i);
        if (lane >= i) thread_ofs += tmp;
    }
    __syncthreads();

    if (lane == 31) {
        locOffset[warp + 1] = thread_ofs;
        __threadfence();
        if (warp == 0) {
            flag[0] = 2;
            __threadfence();
            flag[1] = 1;
            __threadfence();
        } else {
            flag[warp + 1] = 1;
            __threadfence();
        }
    }
    __syncthreads();

    if (warp > 0) {
        if (!lane) {
            int lookback = warp;
            int loc_excl_sum = 0;
            while (lookback > 0) {
                int status;
                do {
                    status = flag[lookback];
                    __threadfence();
                } while (status == 0);
                if (status == 2) {
                    loc_excl_sum += cmpOffset[lookback];
                    __threadfence();
                    break;
                }
                if (status == 1) loc_excl_sum += locOffset[lookback];
                lookback--;
                __threadfence();
            }
            excl_sum = loc_excl_sum;
        }
        __syncthreads();
    }

    if (warp > 0) {
        if (!lane) {
            cmpOffset[warp] = excl_sum;
            __threadfence();
            if (warp == gridDim.x - 1) cmpOffset[warp + 1] = cmpOffset[warp] + locOffset[warp + 1];
            __threadfence();
            flag[warp] = 2;
            __threadfence();
        }
    }
    __syncthreads();

    if (!lane) base_idx = excl_sum + rate_ofs;
    __syncthreads();

    unsigned int base_cmp_byte_ofs = base_idx;
    unsigned int cmp_byte_ofs;
    unsigned int tmp_byte_ofs = 0;
    unsigned int cur_byte_ofs = 0;
    for (int j = 0; j < block_num; j++) {
        int chunk_idx_start = j * DBLOCK;

        tmp_byte_ofs = (fixed_rate[j]) ? (SIGN_BYTES + fixed_rate[j] * SIGN_BYTES) : 0;
        #pragma unroll 5
        for (int i = 1; i < 32; i <<= 1) {
            int tmp = __shfl_up_sync(0xffffffff, tmp_byte_ofs, i);
            if (lane >= i) tmp_byte_ofs += tmp;
        }
        unsigned int prev_thread = __shfl_up_sync(0xffffffff, tmp_byte_ofs, 1);
        if (!lane) cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs;
        else cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs + prev_thread;

        if (fixed_rate[j]) {
            for (int sw = 0; sw < SIGN_WORDS; sw++) {
                unsigned int sf = sign_flag[j * SIGN_WORDS + sw];
                tmp_char.x = 0xff & (sf >> 24);
                tmp_char.y = 0xff & (sf >> 16);
                tmp_char.z = 0xff & (sf >> 8);
                tmp_char.w = 0xff & sf;
                reinterpret_cast<uchar4*>(cmpData)[cmp_byte_ofs / 4] = tmp_char;
                cmp_byte_ofs += 4;
            }

            int mask = 1;
            for (int i = 0; i < fixed_rate[j]; i++) {
                for (int g = 0; g < DBLOCK; g += 32) {
                    tmp_char.x = (((absQuant[chunk_idx_start + g + 0] & mask) >> i) << 7) |
                                 (((absQuant[chunk_idx_start + g + 1] & mask) >> i) << 6) |
                                 (((absQuant[chunk_idx_start + g + 2] & mask) >> i) << 5) |
                                 (((absQuant[chunk_idx_start + g + 3] & mask) >> i) << 4) |
                                 (((absQuant[chunk_idx_start + g + 4] & mask) >> i) << 3) |
                                 (((absQuant[chunk_idx_start + g + 5] & mask) >> i) << 2) |
                                 (((absQuant[chunk_idx_start + g + 6] & mask) >> i) << 1) |
                                 (((absQuant[chunk_idx_start + g + 7] & mask) >> i) << 0);

                    tmp_char.y = (((absQuant[chunk_idx_start + g + 8] & mask) >> i) << 7) |
                                 (((absQuant[chunk_idx_start + g + 9] & mask) >> i) << 6) |
                                 (((absQuant[chunk_idx_start + g + 10] & mask) >> i) << 5) |
                                 (((absQuant[chunk_idx_start + g + 11] & mask) >> i) << 4) |
                                 (((absQuant[chunk_idx_start + g + 12] & mask) >> i) << 3) |
                                 (((absQuant[chunk_idx_start + g + 13] & mask) >> i) << 2) |
                                 (((absQuant[chunk_idx_start + g + 14] & mask) >> i) << 1) |
                                 (((absQuant[chunk_idx_start + g + 15] & mask) >> i) << 0);

                    tmp_char.z = (((absQuant[chunk_idx_start + g + 16] & mask) >> i) << 7) |
                                 (((absQuant[chunk_idx_start + g + 17] & mask) >> i) << 6) |
                                 (((absQuant[chunk_idx_start + g + 18] & mask) >> i) << 5) |
                                 (((absQuant[chunk_idx_start + g + 19] & mask) >> i) << 4) |
                                 (((absQuant[chunk_idx_start + g + 20] & mask) >> i) << 3) |
                                 (((absQuant[chunk_idx_start + g + 21] & mask) >> i) << 2) |
                                 (((absQuant[chunk_idx_start + g + 22] & mask) >> i) << 1) |
                                 (((absQuant[chunk_idx_start + g + 23] & mask) >> i) << 0);

                    tmp_char.w = (((absQuant[chunk_idx_start + g + 24] & mask) >> i) << 7) |
                                 (((absQuant[chunk_idx_start + g + 25] & mask) >> i) << 6) |
                                 (((absQuant[chunk_idx_start + g + 26] & mask) >> i) << 5) |
                                 (((absQuant[chunk_idx_start + g + 27] & mask) >> i) << 4) |
                                 (((absQuant[chunk_idx_start + g + 28] & mask) >> i) << 3) |
                                 (((absQuant[chunk_idx_start + g + 29] & mask) >> i) << 2) |
                                 (((absQuant[chunk_idx_start + g + 30] & mask) >> i) << 1) |
                                 (((absQuant[chunk_idx_start + g + 31] & mask) >> i) << 0);

                    reinterpret_cast<uchar4*>(cmpData)[cmp_byte_ofs / 4] = tmp_char;
                    cmp_byte_ofs += 4;
                }
                mask <<= 1;
            }
        }

        cur_byte_ofs += __shfl_sync(0xffffffff, tmp_byte_ofs, 31);
    }
}

template <int DBLOCK>
__global__ void cuSZp_decompress_kernel_1D_plain_f32_block_size_N(
    float* const __restrict__ decData, const unsigned char* const __restrict__ cmpData,
    volatile unsigned int* const __restrict__ cmpOffset,
    volatile unsigned int* const __restrict__ locOffset,
    volatile int* const __restrict__ flag, const float eb, const size_t nbEle)
{
    __shared__ unsigned int excl_sum;
    __shared__ unsigned int base_idx;

    constexpr int SIGN_WORDS = DBLOCK / 32;
    constexpr int SIGN_BYTES = DBLOCK / 8;
    constexpr int block_num = thread_chunk / DBLOCK;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int idx = bid * blockDim.x + tid;
    const int lane = idx & 0x1f;
    const int warp = idx >> 5;
    const int rate_ofs =
        (nbEle + (size_t)tblock_size * thread_chunk - 1) /
        ((size_t)tblock_size * thread_chunk) * ((size_t)tblock_size * thread_chunk) / DBLOCK;

    if (!tid) {
        excl_sum = 0;
        base_idx = 0;
    }
    __syncthreads();

    int base_start_idx;
    int base_block_start_idx, base_block_end_idx;
    int block_idx;
    int absQuant[DBLOCK];
    int currQuant, lorenQuant, prevQuant;
    int fixed_rate[block_num];
    unsigned int thread_ofs = 0;
    uchar4 tmp_char;
    float4 dec_buffer;

    for (int j = 0; j < block_num; j++) {
        block_idx = warp * (32 * block_num) + j * 32 + lane;
        fixed_rate[j] = (int)cmpData[block_idx];
        thread_ofs += (fixed_rate[j]) ? (SIGN_BYTES + fixed_rate[j] * SIGN_BYTES) : 0;
        __syncthreads();
    }

    #pragma unroll 5
    for (int i = 1; i < 32; i <<= 1) {
        int tmp = __shfl_up_sync(0xffffffff, thread_ofs, i);
        if (lane >= i) thread_ofs += tmp;
    }
    __syncthreads();

    if (lane == 31) {
        locOffset[warp + 1] = thread_ofs;
        __threadfence();
        if (warp == 0) {
            flag[0] = 2;
            __threadfence();
            flag[1] = 1;
            __threadfence();
        } else {
            flag[warp + 1] = 1;
            __threadfence();
        }
    }
    __syncthreads();

    if (warp > 0) {
        if (!lane) {
            int lookback = warp;
            int loc_excl_sum = 0;
            while (lookback > 0) {
                int status;
                do {
                    status = flag[lookback];
                    __threadfence();
                } while (status == 0);
                if (status == 2) {
                    loc_excl_sum += cmpOffset[lookback];
                    __threadfence();
                    break;
                }
                if (status == 1) loc_excl_sum += locOffset[lookback];
                lookback--;
                __threadfence();
            }
            excl_sum = loc_excl_sum;
        }
        __syncthreads();
    }

    if (warp > 0) {
        if (!lane) cmpOffset[warp] = excl_sum;
        __threadfence();
        if (!lane) flag[warp] = 2;
        __threadfence();
    }
    __syncthreads();

    if (!lane) base_idx = excl_sum + rate_ofs;
    __syncthreads();

    unsigned int base_cmp_byte_ofs = base_idx;
    unsigned int cmp_byte_ofs;
    unsigned int tmp_byte_ofs = 0;
    unsigned int cur_byte_ofs = 0;
    base_start_idx = warp * thread_chunk * 32;
    for (int j = 0; j < block_num; j++) {
        base_block_start_idx = base_start_idx + j * (32 * DBLOCK) + lane * DBLOCK;
        base_block_end_idx = base_block_start_idx + DBLOCK;
        unsigned int sign_flags[SIGN_WORDS];
        for (int sw = 0; sw < SIGN_WORDS; sw++) sign_flags[sw] = 0;

        tmp_byte_ofs = (fixed_rate[j]) ? (SIGN_BYTES + fixed_rate[j] * SIGN_BYTES) : 0;
        #pragma unroll 5
        for (int i = 1; i < 32; i <<= 1) {
            int tmp = __shfl_up_sync(0xffffffff, tmp_byte_ofs, i);
            if (lane >= i) tmp_byte_ofs += tmp;
        }
        unsigned int prev_thread = __shfl_up_sync(0xffffffff, tmp_byte_ofs, 1);
        if (!lane) cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs;
        else cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs + prev_thread;

        if (fixed_rate[j]) {
            for (int sw = 0; sw < SIGN_WORDS; sw++) {
                tmp_char = reinterpret_cast<const uchar4*>(cmpData)[cmp_byte_ofs / 4];
                sign_flags[sw] = (0xff000000 & (tmp_char.x << 24)) |
                                 (0x00ff0000 & (tmp_char.y << 16)) |
                                 (0x0000ff00 & (tmp_char.z << 8)) |
                                 (0x000000ff & tmp_char.w);
                cmp_byte_ofs += 4;
            }

            for (int i = 0; i < DBLOCK; i++) absQuant[i] = 0;
            for (int i = 0; i < fixed_rate[j]; i++) {
                for (int g = 0; g < DBLOCK; g += 32) {
                    tmp_char = reinterpret_cast<const uchar4*>(cmpData)[cmp_byte_ofs / 4];
                    absQuant[g + 0]  |= ((tmp_char.x >> 7) & 0x01) << i;
                    absQuant[g + 1]  |= ((tmp_char.x >> 6) & 0x01) << i;
                    absQuant[g + 2]  |= ((tmp_char.x >> 5) & 0x01) << i;
                    absQuant[g + 3]  |= ((tmp_char.x >> 4) & 0x01) << i;
                    absQuant[g + 4]  |= ((tmp_char.x >> 3) & 0x01) << i;
                    absQuant[g + 5]  |= ((tmp_char.x >> 2) & 0x01) << i;
                    absQuant[g + 6]  |= ((tmp_char.x >> 1) & 0x01) << i;
                    absQuant[g + 7]  |= ((tmp_char.x >> 0) & 0x01) << i;
                    absQuant[g + 8]  |= ((tmp_char.y >> 7) & 0x01) << i;
                    absQuant[g + 9]  |= ((tmp_char.y >> 6) & 0x01) << i;
                    absQuant[g + 10] |= ((tmp_char.y >> 5) & 0x01) << i;
                    absQuant[g + 11] |= ((tmp_char.y >> 4) & 0x01) << i;
                    absQuant[g + 12] |= ((tmp_char.y >> 3) & 0x01) << i;
                    absQuant[g + 13] |= ((tmp_char.y >> 2) & 0x01) << i;
                    absQuant[g + 14] |= ((tmp_char.y >> 1) & 0x01) << i;
                    absQuant[g + 15] |= ((tmp_char.y >> 0) & 0x01) << i;
                    absQuant[g + 16] |= ((tmp_char.z >> 7) & 0x01) << i;
                    absQuant[g + 17] |= ((tmp_char.z >> 6) & 0x01) << i;
                    absQuant[g + 18] |= ((tmp_char.z >> 5) & 0x01) << i;
                    absQuant[g + 19] |= ((tmp_char.z >> 4) & 0x01) << i;
                    absQuant[g + 20] |= ((tmp_char.z >> 3) & 0x01) << i;
                    absQuant[g + 21] |= ((tmp_char.z >> 2) & 0x01) << i;
                    absQuant[g + 22] |= ((tmp_char.z >> 1) & 0x01) << i;
                    absQuant[g + 23] |= ((tmp_char.z >> 0) & 0x01) << i;
                    absQuant[g + 24] |= ((tmp_char.w >> 7) & 0x01) << i;
                    absQuant[g + 25] |= ((tmp_char.w >> 6) & 0x01) << i;
                    absQuant[g + 26] |= ((tmp_char.w >> 5) & 0x01) << i;
                    absQuant[g + 27] |= ((tmp_char.w >> 4) & 0x01) << i;
                    absQuant[g + 28] |= ((tmp_char.w >> 3) & 0x01) << i;
                    absQuant[g + 29] |= ((tmp_char.w >> 2) & 0x01) << i;
                    absQuant[g + 30] |= ((tmp_char.w >> 1) & 0x01) << i;
                    absQuant[g + 31] |= ((tmp_char.w >> 0) & 0x01) << i;
                    cmp_byte_ofs += 4;
                }
            }

            prevQuant = 0;
            if (base_block_end_idx < nbEle) {
                for (int i = 0; i < DBLOCK; i += 4) {
                    int sw0 = i / 32, sb0 = i % 32;
                    lorenQuant = sign_flags[sw0] & (1 << (31 - sb0)) ? -absQuant[i] : absQuant[i];
                    currQuant = lorenQuant + prevQuant;
                    prevQuant = currQuant;
                    dec_buffer.x = currQuant * eb * 2;

                    int sw1 = (i + 1) / 32, sb1 = (i + 1) % 32;
                    lorenQuant =
                        sign_flags[sw1] & (1 << (31 - sb1)) ? -absQuant[i + 1] : absQuant[i + 1];
                    currQuant = lorenQuant + prevQuant;
                    prevQuant = currQuant;
                    dec_buffer.y = currQuant * eb * 2;

                    int sw2 = (i + 2) / 32, sb2 = (i + 2) % 32;
                    lorenQuant =
                        sign_flags[sw2] & (1 << (31 - sb2)) ? -absQuant[i + 2] : absQuant[i + 2];
                    currQuant = lorenQuant + prevQuant;
                    prevQuant = currQuant;
                    dec_buffer.z = currQuant * eb * 2;

                    int sw3 = (i + 3) / 32, sb3 = (i + 3) % 32;
                    lorenQuant =
                        sign_flags[sw3] & (1 << (31 - sb3)) ? -absQuant[i + 3] : absQuant[i + 3];
                    currQuant = lorenQuant + prevQuant;
                    prevQuant = currQuant;
                    dec_buffer.w = currQuant * eb * 2;

                    reinterpret_cast<float4*>(decData)[(base_block_start_idx + i) / 4] = dec_buffer;
                }
            } else {
                for (int i = 0; i < DBLOCK; i++) {
                    int sw_i = i / 32, sb_i = i % 32;
                    lorenQuant =
                        sign_flags[sw_i] & (1 << (31 - sb_i)) ? -absQuant[i] : absQuant[i];
                    currQuant = lorenQuant + prevQuant;
                    prevQuant = currQuant;
                    if (base_block_start_idx + i < nbEle) {
                        decData[base_block_start_idx + i] = currQuant * eb * 2;
                    }
                }
            }
        }

        cur_byte_ofs += __shfl_sync(0xffffffff, tmp_byte_ofs, 31);
    }
}

template __global__ void cuSZp_compress_kernel_1D_plain_f32_block_size_N<32>(
    const float* const __restrict__, unsigned char* const __restrict__,
    volatile unsigned int* const __restrict__, volatile unsigned int* const __restrict__,
    volatile int* const __restrict__, const float, const size_t);
template __global__ void cuSZp_compress_kernel_1D_plain_f32_block_size_N<64>(
    const float* const __restrict__, unsigned char* const __restrict__,
    volatile unsigned int* const __restrict__, volatile unsigned int* const __restrict__,
    volatile int* const __restrict__, const float, const size_t);
template __global__ void cuSZp_compress_kernel_1D_plain_f32_block_size_N<128>(
    const float* const __restrict__, unsigned char* const __restrict__,
    volatile unsigned int* const __restrict__, volatile unsigned int* const __restrict__,
    volatile int* const __restrict__, const float, const size_t);
template __global__ void cuSZp_compress_kernel_1D_plain_f32_block_size_N<256>(
    const float* const __restrict__, unsigned char* const __restrict__,
    volatile unsigned int* const __restrict__, volatile unsigned int* const __restrict__,
    volatile int* const __restrict__, const float, const size_t);

template __global__ void cuSZp_decompress_kernel_1D_plain_f32_block_size_N<32>(
    float* const __restrict__, const unsigned char* const __restrict__,
    volatile unsigned int* const __restrict__, volatile unsigned int* const __restrict__,
    volatile int* const __restrict__, const float, const size_t);
template __global__ void cuSZp_decompress_kernel_1D_plain_f32_block_size_N<64>(
    float* const __restrict__, const unsigned char* const __restrict__,
    volatile unsigned int* const __restrict__, volatile unsigned int* const __restrict__,
    volatile int* const __restrict__, const float, const size_t);
template __global__ void cuSZp_decompress_kernel_1D_plain_f32_block_size_N<128>(
    float* const __restrict__, const unsigned char* const __restrict__,
    volatile unsigned int* const __restrict__, volatile unsigned int* const __restrict__,
    volatile int* const __restrict__, const float, const size_t);
template __global__ void cuSZp_decompress_kernel_1D_plain_f32_block_size_N<256>(
    float* const __restrict__, const unsigned char* const __restrict__,
    volatile unsigned int* const __restrict__, volatile unsigned int* const __restrict__,
    volatile int* const __restrict__, const float, const size_t);

static __device__ inline unsigned char pack_group_byte_block_size_N(const int* absQuant,
                                                                    int group_start, int bit,
                                                                    bool skip_first)
{
    unsigned char packed = 0;
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        if (skip_first && group_start + k == 0) continue;
        packed |= (unsigned char)(((absQuant[group_start + k] >> bit) & 0x1) << (7 - k));
    }
    return packed;
}

static __device__ inline void unpack_group_byte_block_size_N(int* absQuant, int group_start,
                                                             int bit, unsigned char packed,
                                                             bool skip_first)
{
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        if (skip_first && group_start + k == 0) continue;
        absQuant[group_start + k] |= ((packed >> (7 - k)) & 0x1) << bit;
    }
}

template <int DBLOCK>
__global__ void cuSZp_compress_kernel_1D_outlier_f32_block_size_N(
    const float* const __restrict__ oriData, unsigned char* const __restrict__ cmpData,
    volatile unsigned int* const __restrict__ cmpOffset,
    volatile unsigned int* const __restrict__ locOffset,
    volatile int* const __restrict__ flag, const float eb, const size_t nbEle)
{
    __shared__ unsigned int excl_sum;
    __shared__ unsigned int base_idx;

    constexpr int SIGN_WORDS = DBLOCK / 32;
    constexpr int SIGN_BYTES = DBLOCK / 8;
    constexpr int block_num = thread_chunk / DBLOCK;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int idx = bid * blockDim.x + tid;
    const int lane = idx & 0x1f;
    const int warp = idx >> 5;
    const int rate_ofs =
        (nbEle + (size_t)tblock_size * thread_chunk - 1) /
        ((size_t)tblock_size * thread_chunk) * ((size_t)tblock_size * thread_chunk) / DBLOCK;
    const float recipPrecision = 0.5f / eb;

    if (!tid) {
        excl_sum = 0;
        base_idx = 0;
    }
    __syncthreads();

    int base_start_idx;
    int base_block_start_idx, base_block_end_idx;
    int quant_chunk_idx;
    int block_idx;
    int currQuant, lorenQuant, prevQuant;
    int absQuant[thread_chunk];
    unsigned int sign_flag[block_num * SIGN_WORDS];
    int fixed_rate[block_num];
    unsigned int thread_ofs = 0;
    float4 tmp_buffer;

    base_start_idx = warp * thread_chunk * 32;
    for (int j = 0; j < block_num; j++) {
        base_block_start_idx = base_start_idx + j * (32 * DBLOCK) + lane * DBLOCK;
        base_block_end_idx = base_block_start_idx + DBLOCK;
        for (int sw = 0; sw < SIGN_WORDS; sw++) sign_flag[j * SIGN_WORDS + sw] = 0;
        block_idx = base_block_start_idx / DBLOCK;
        prevQuant = 0;
        int maxQuant = 0;
        int maxQuan2 = 0;
        int outlier = 0;

        if (base_block_end_idx < nbEle) {
            for (int i = base_block_start_idx; i < base_block_end_idx; i += 4) {
                tmp_buffer = reinterpret_cast<const float4*>(oriData)[i / 4];
                quant_chunk_idx = j * DBLOCK + (i - base_block_start_idx);
                int local_i = i - base_block_start_idx;

                currQuant = quantization_block_size_N(tmp_buffer.x, recipPrecision);
                lorenQuant = currQuant - prevQuant;
                prevQuant = currQuant;
                sign_flag[j * SIGN_WORDS + local_i / 32] |=
                    (lorenQuant < 0) << (31 - (local_i % 32));
                absQuant[quant_chunk_idx] = abs(lorenQuant);
                maxQuant = maxQuant > absQuant[quant_chunk_idx] ? maxQuant
                                                                 : absQuant[quant_chunk_idx];
                if (local_i == 0) outlier = absQuant[quant_chunk_idx];
                else maxQuan2 = maxQuan2 > absQuant[quant_chunk_idx] ? maxQuan2
                                                                      : absQuant[quant_chunk_idx];

                currQuant = quantization_block_size_N(tmp_buffer.y, recipPrecision);
                lorenQuant = currQuant - prevQuant;
                prevQuant = currQuant;
                sign_flag[j * SIGN_WORDS + (local_i + 1) / 32] |=
                    (lorenQuant < 0) << (31 - ((local_i + 1) % 32));
                absQuant[quant_chunk_idx + 1] = abs(lorenQuant);
                maxQuant = maxQuant > absQuant[quant_chunk_idx + 1]
                               ? maxQuant
                               : absQuant[quant_chunk_idx + 1];
                maxQuan2 = maxQuan2 > absQuant[quant_chunk_idx + 1]
                               ? maxQuan2
                               : absQuant[quant_chunk_idx + 1];

                currQuant = quantization_block_size_N(tmp_buffer.z, recipPrecision);
                lorenQuant = currQuant - prevQuant;
                prevQuant = currQuant;
                sign_flag[j * SIGN_WORDS + (local_i + 2) / 32] |=
                    (lorenQuant < 0) << (31 - ((local_i + 2) % 32));
                absQuant[quant_chunk_idx + 2] = abs(lorenQuant);
                maxQuant = maxQuant > absQuant[quant_chunk_idx + 2]
                               ? maxQuant
                               : absQuant[quant_chunk_idx + 2];
                maxQuan2 = maxQuan2 > absQuant[quant_chunk_idx + 2]
                               ? maxQuan2
                               : absQuant[quant_chunk_idx + 2];

                currQuant = quantization_block_size_N(tmp_buffer.w, recipPrecision);
                lorenQuant = currQuant - prevQuant;
                prevQuant = currQuant;
                sign_flag[j * SIGN_WORDS + (local_i + 3) / 32] |=
                    (lorenQuant < 0) << (31 - ((local_i + 3) % 32));
                absQuant[quant_chunk_idx + 3] = abs(lorenQuant);
                maxQuant = maxQuant > absQuant[quant_chunk_idx + 3]
                               ? maxQuant
                               : absQuant[quant_chunk_idx + 3];
                maxQuan2 = maxQuan2 > absQuant[quant_chunk_idx + 3]
                               ? maxQuan2
                               : absQuant[quant_chunk_idx + 3];
            }
        } else {
            if (base_block_start_idx >= nbEle) {
                for (int i = 0; i < DBLOCK; i++) absQuant[j * DBLOCK + i] = 0;
            } else {
                int remainbEle = nbEle - base_block_start_idx;
                for (int i = 0; i < remainbEle; i++) {
                    quant_chunk_idx = j * DBLOCK + i;
                    currQuant =
                        quantization_block_size_N(oriData[base_block_start_idx + i], recipPrecision);
                    lorenQuant = currQuant - prevQuant;
                    prevQuant = currQuant;
                    sign_flag[j * SIGN_WORDS + i / 32] |= (lorenQuant < 0) << (31 - (i % 32));
                    absQuant[quant_chunk_idx] = abs(lorenQuant);
                    maxQuant = maxQuant > absQuant[quant_chunk_idx] ? maxQuant
                                                                     : absQuant[quant_chunk_idx];
                    if (i == 0) outlier = absQuant[quant_chunk_idx];
                    else maxQuan2 = maxQuan2 > absQuant[quant_chunk_idx] ? maxQuan2
                                                                          : absQuant[quant_chunk_idx];
                }
                for (int i = remainbEle; i < DBLOCK; i++) absQuant[j * DBLOCK + i] = 0;
            }
        }

        int fr1 = get_bit_num_block_size_N(maxQuant);
        int fr2 = get_bit_num_block_size_N(maxQuan2);
        int outlier_byte_num = (get_bit_num_block_size_N(outlier) + 7) / 8;
        int temp_ofs1 = fr1 ? SIGN_BYTES + fr1 * SIGN_BYTES : 0;
        int temp_ofs2 = SIGN_BYTES + outlier_byte_num + fr2 * SIGN_BYTES;
        int temp_rate = 0;
        if (temp_ofs1 <= temp_ofs2) {
            thread_ofs += temp_ofs1;
            temp_rate = fr1;
        } else {
            thread_ofs += temp_ofs2;
            temp_rate = fr2 | 0x80 | ((outlier_byte_num - 1) << 5);
        }

        fixed_rate[j] = temp_rate;
        cmpData[block_idx] = (unsigned char)fixed_rate[j];
        __syncthreads();
    }

    #pragma unroll 5
    for (int i = 1; i < 32; i <<= 1) {
        int tmp = __shfl_up_sync(0xffffffff, thread_ofs, i);
        if (lane >= i) thread_ofs += tmp;
    }
    __syncthreads();

    if (lane == 31) {
        locOffset[warp + 1] = thread_ofs;
        __threadfence();
        if (warp == 0) {
            flag[0] = 2;
            __threadfence();
            flag[1] = 1;
            __threadfence();
        } else {
            flag[warp + 1] = 1;
            __threadfence();
        }
    }
    __syncthreads();

    if (warp > 0) {
        if (!lane) {
            int lookback = warp;
            int loc_excl_sum = 0;
            while (lookback > 0) {
                int status;
                do {
                    status = flag[lookback];
                    __threadfence();
                } while (status == 0);
                if (status == 2) {
                    loc_excl_sum += cmpOffset[lookback];
                    __threadfence();
                    break;
                }
                if (status == 1) loc_excl_sum += locOffset[lookback];
                lookback--;
                __threadfence();
            }
            excl_sum = loc_excl_sum;
        }
        __syncthreads();
    }

    if (warp > 0) {
        if (!lane) {
            cmpOffset[warp] = excl_sum;
            __threadfence();
            if (warp == gridDim.x - 1) cmpOffset[warp + 1] = cmpOffset[warp] + locOffset[warp + 1];
            __threadfence();
            flag[warp] = 2;
            __threadfence();
        }
    }
    __syncthreads();

    if (!lane) base_idx = excl_sum + rate_ofs;
    __syncthreads();

    unsigned int base_cmp_byte_ofs = base_idx;
    unsigned int cmp_byte_ofs;
    unsigned int tmp_byte_ofs = 0;
    unsigned int cur_byte_ofs = 0;
    for (int j = 0; j < block_num; j++) {
        int encoding_selection = fixed_rate[j] >> 7;
        int outlier_byte_num = encoding_selection ? (((fixed_rate[j] & 0x60) >> 5) + 1) : 0;
        int payload_rate = fixed_rate[j] & 0x1f;
        int chunk_idx_start = j * DBLOCK;

        if (!encoding_selection) tmp_byte_ofs = payload_rate ? (SIGN_BYTES + payload_rate * SIGN_BYTES) : 0;
        else tmp_byte_ofs = SIGN_BYTES + outlier_byte_num + payload_rate * SIGN_BYTES;
        #pragma unroll 5
        for (int i = 1; i < 32; i <<= 1) {
            int tmp = __shfl_up_sync(0xffffffff, tmp_byte_ofs, i);
            if (lane >= i) tmp_byte_ofs += tmp;
        }
        unsigned int prev_thread = __shfl_up_sync(0xffffffff, tmp_byte_ofs, 1);
        if (!lane) cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs;
        else cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs + prev_thread;

        if (encoding_selection) {
            int outlier_buffer = absQuant[chunk_idx_start];
            for (int i = 0; i < outlier_byte_num; i++) {
                cmpData[cmp_byte_ofs++] = (unsigned char)(outlier_buffer & 0xff);
                outlier_buffer >>= 8;
            }
        }

        if (payload_rate || encoding_selection) {
            for (int sw = 0; sw < SIGN_WORDS; sw++) {
                unsigned int sf = sign_flag[j * SIGN_WORDS + sw];
                cmpData[cmp_byte_ofs++] = 0xff & (sf >> 24);
                cmpData[cmp_byte_ofs++] = 0xff & (sf >> 16);
                cmpData[cmp_byte_ofs++] = 0xff & (sf >> 8);
                cmpData[cmp_byte_ofs++] = 0xff & sf;
            }
        }

        if (payload_rate) {
            for (int i = 0; i < payload_rate; i++) {
                for (int g = 0; g < DBLOCK; g += 32) {
                    uchar4 tmp_char;
                    tmp_char.x = pack_group_byte_block_size_N(absQuant + chunk_idx_start, g + 0, i,
                                                              encoding_selection);
                    tmp_char.y = pack_group_byte_block_size_N(absQuant + chunk_idx_start, g + 8, i,
                                                              false);
                    tmp_char.z = pack_group_byte_block_size_N(absQuant + chunk_idx_start, g + 16, i,
                                                              false);
                    tmp_char.w = pack_group_byte_block_size_N(absQuant + chunk_idx_start, g + 24, i,
                                                              false);
                    cmpData[cmp_byte_ofs++] = tmp_char.x;
                    cmpData[cmp_byte_ofs++] = tmp_char.y;
                    cmpData[cmp_byte_ofs++] = tmp_char.z;
                    cmpData[cmp_byte_ofs++] = tmp_char.w;
                }
            }
        }

        cur_byte_ofs += __shfl_sync(0xffffffff, tmp_byte_ofs, 31);
    }
}

template <int DBLOCK>
__global__ void cuSZp_decompress_kernel_1D_outlier_f32_block_size_N(
    float* const __restrict__ decData, const unsigned char* const __restrict__ cmpData,
    volatile unsigned int* const __restrict__ cmpOffset,
    volatile unsigned int* const __restrict__ locOffset,
    volatile int* const __restrict__ flag, const float eb, const size_t nbEle)
{
    __shared__ unsigned int excl_sum;
    __shared__ unsigned int base_idx;

    constexpr int SIGN_WORDS = DBLOCK / 32;
    constexpr int SIGN_BYTES = DBLOCK / 8;
    constexpr int block_num = thread_chunk / DBLOCK;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int idx = bid * blockDim.x + tid;
    const int lane = idx & 0x1f;
    const int warp = idx >> 5;
    const int rate_ofs =
        (nbEle + (size_t)tblock_size * thread_chunk - 1) /
        ((size_t)tblock_size * thread_chunk) * ((size_t)tblock_size * thread_chunk) / DBLOCK;

    if (!tid) {
        excl_sum = 0;
        base_idx = 0;
    }
    __syncthreads();

    int base_start_idx;
    int base_block_start_idx, base_block_end_idx;
    int block_idx;
    int absQuant[DBLOCK];
    int currQuant, lorenQuant, prevQuant;
    int fixed_rate[block_num];
    unsigned int thread_ofs = 0;
    float4 dec_buffer;

    for (int j = 0; j < block_num; j++) {
        block_idx = warp * (32 * block_num) + j * 32 + lane;
        fixed_rate[j] = (int)cmpData[block_idx];
        int encoding_selection = fixed_rate[j] >> 7;
        int outlier_byte_num = encoding_selection ? (((fixed_rate[j] & 0x60) >> 5) + 1) : 0;
        int payload_rate = fixed_rate[j] & 0x1f;
        if (!encoding_selection) thread_ofs += payload_rate ? (SIGN_BYTES + payload_rate * SIGN_BYTES) : 0;
        else thread_ofs += SIGN_BYTES + outlier_byte_num + payload_rate * SIGN_BYTES;
        __syncthreads();
    }

    #pragma unroll 5
    for (int i = 1; i < 32; i <<= 1) {
        int tmp = __shfl_up_sync(0xffffffff, thread_ofs, i);
        if (lane >= i) thread_ofs += tmp;
    }
    __syncthreads();

    if (lane == 31) {
        locOffset[warp + 1] = thread_ofs;
        __threadfence();
        if (warp == 0) {
            flag[0] = 2;
            __threadfence();
            flag[1] = 1;
            __threadfence();
        } else {
            flag[warp + 1] = 1;
            __threadfence();
        }
    }
    __syncthreads();

    if (warp > 0) {
        if (!lane) {
            int lookback = warp;
            int loc_excl_sum = 0;
            while (lookback > 0) {
                int status;
                do {
                    status = flag[lookback];
                    __threadfence();
                } while (status == 0);
                if (status == 2) {
                    loc_excl_sum += cmpOffset[lookback];
                    __threadfence();
                    break;
                }
                if (status == 1) loc_excl_sum += locOffset[lookback];
                lookback--;
                __threadfence();
            }
            excl_sum = loc_excl_sum;
        }
        __syncthreads();
    }

    if (warp > 0) {
        if (!lane) cmpOffset[warp] = excl_sum;
        __threadfence();
        if (!lane) flag[warp] = 2;
        __threadfence();
    }
    __syncthreads();

    if (!lane) base_idx = excl_sum + rate_ofs;
    __syncthreads();

    unsigned int base_cmp_byte_ofs = base_idx;
    unsigned int cmp_byte_ofs;
    unsigned int tmp_byte_ofs = 0;
    unsigned int cur_byte_ofs = 0;
    base_start_idx = warp * thread_chunk * 32;
    for (int j = 0; j < block_num; j++) {
        int encoding_selection = fixed_rate[j] >> 7;
        int outlier_byte_num = encoding_selection ? (((fixed_rate[j] & 0x60) >> 5) + 1) : 0;
        int payload_rate = fixed_rate[j] & 0x1f;
        unsigned int sign_flags[SIGN_WORDS];
        base_block_start_idx = base_start_idx + j * (32 * DBLOCK) + lane * DBLOCK;
        base_block_end_idx = base_block_start_idx + DBLOCK;

        if (!encoding_selection) tmp_byte_ofs = payload_rate ? (SIGN_BYTES + payload_rate * SIGN_BYTES) : 0;
        else tmp_byte_ofs = SIGN_BYTES + outlier_byte_num + payload_rate * SIGN_BYTES;
        #pragma unroll 5
        for (int i = 1; i < 32; i <<= 1) {
            int tmp = __shfl_up_sync(0xffffffff, tmp_byte_ofs, i);
            if (lane >= i) tmp_byte_ofs += tmp;
        }
        unsigned int prev_thread = __shfl_up_sync(0xffffffff, tmp_byte_ofs, 1);
        if (!lane) cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs;
        else cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs + prev_thread;

        if (payload_rate || encoding_selection) {
            for (int i = 0; i < DBLOCK; i++) absQuant[i] = 0;
            for (int sw = 0; sw < SIGN_WORDS; sw++) sign_flags[sw] = 0;

            if (encoding_selection) {
                int outlier_buffer = 0;
                for (int i = 0; i < outlier_byte_num; i++) {
                    outlier_buffer |= ((int)cmpData[cmp_byte_ofs++]) << (8 * i);
                }
                absQuant[0] = outlier_buffer;
            }

            for (int sw = 0; sw < SIGN_WORDS; sw++) {
                sign_flags[sw] = (0xff000000 & (cmpData[cmp_byte_ofs++] << 24)) |
                                 (0x00ff0000 & (cmpData[cmp_byte_ofs++] << 16)) |
                                 (0x0000ff00 & (cmpData[cmp_byte_ofs++] << 8)) |
                                 (0x000000ff & cmpData[cmp_byte_ofs++]);
            }

            if (payload_rate) {
                for (int i = 0; i < payload_rate; i++) {
                    for (int g = 0; g < DBLOCK; g += 32) {
                        unsigned char bx = cmpData[cmp_byte_ofs++];
                        unsigned char by = cmpData[cmp_byte_ofs++];
                        unsigned char bz = cmpData[cmp_byte_ofs++];
                        unsigned char bw = cmpData[cmp_byte_ofs++];
                        unpack_group_byte_block_size_N(absQuant, g + 0, i, bx, encoding_selection);
                        unpack_group_byte_block_size_N(absQuant, g + 8, i, by, false);
                        unpack_group_byte_block_size_N(absQuant, g + 16, i, bz, false);
                        unpack_group_byte_block_size_N(absQuant, g + 24, i, bw, false);
                    }
                }
            }

            prevQuant = 0;
            if (base_block_end_idx < nbEle) {
                for (int i = 0; i < DBLOCK; i += 4) {
                    int sw0 = i / 32, sb0 = i % 32;
                    lorenQuant = sign_flags[sw0] & (1 << (31 - sb0)) ? -absQuant[i] : absQuant[i];
                    currQuant = lorenQuant + prevQuant;
                    prevQuant = currQuant;
                    dec_buffer.x = currQuant * eb * 2;

                    int sw1 = (i + 1) / 32, sb1 = (i + 1) % 32;
                    lorenQuant =
                        sign_flags[sw1] & (1 << (31 - sb1)) ? -absQuant[i + 1] : absQuant[i + 1];
                    currQuant = lorenQuant + prevQuant;
                    prevQuant = currQuant;
                    dec_buffer.y = currQuant * eb * 2;

                    int sw2 = (i + 2) / 32, sb2 = (i + 2) % 32;
                    lorenQuant =
                        sign_flags[sw2] & (1 << (31 - sb2)) ? -absQuant[i + 2] : absQuant[i + 2];
                    currQuant = lorenQuant + prevQuant;
                    prevQuant = currQuant;
                    dec_buffer.z = currQuant * eb * 2;

                    int sw3 = (i + 3) / 32, sb3 = (i + 3) % 32;
                    lorenQuant =
                        sign_flags[sw3] & (1 << (31 - sb3)) ? -absQuant[i + 3] : absQuant[i + 3];
                    currQuant = lorenQuant + prevQuant;
                    prevQuant = currQuant;
                    dec_buffer.w = currQuant * eb * 2;

                    reinterpret_cast<float4*>(decData)[(base_block_start_idx + i) / 4] = dec_buffer;
                }
            } else {
                for (int i = 0; i < DBLOCK; i++) {
                    int sw_i = i / 32, sb_i = i % 32;
                    lorenQuant =
                        sign_flags[sw_i] & (1 << (31 - sb_i)) ? -absQuant[i] : absQuant[i];
                    currQuant = lorenQuant + prevQuant;
                    prevQuant = currQuant;
                    if (base_block_start_idx + i < nbEle) {
                        decData[base_block_start_idx + i] = currQuant * eb * 2;
                    }
                }
            }
        }

        cur_byte_ofs += __shfl_sync(0xffffffff, tmp_byte_ofs, 31);
    }
}

template __global__ void cuSZp_compress_kernel_1D_outlier_f32_block_size_N<32>(
    const float* const __restrict__, unsigned char* const __restrict__,
    volatile unsigned int* const __restrict__, volatile unsigned int* const __restrict__,
    volatile int* const __restrict__, const float, const size_t);
template __global__ void cuSZp_compress_kernel_1D_outlier_f32_block_size_N<64>(
    const float* const __restrict__, unsigned char* const __restrict__,
    volatile unsigned int* const __restrict__, volatile unsigned int* const __restrict__,
    volatile int* const __restrict__, const float, const size_t);
template __global__ void cuSZp_compress_kernel_1D_outlier_f32_block_size_N<128>(
    const float* const __restrict__, unsigned char* const __restrict__,
    volatile unsigned int* const __restrict__, volatile unsigned int* const __restrict__,
    volatile int* const __restrict__, const float, const size_t);
template __global__ void cuSZp_compress_kernel_1D_outlier_f32_block_size_N<256>(
    const float* const __restrict__, unsigned char* const __restrict__,
    volatile unsigned int* const __restrict__, volatile unsigned int* const __restrict__,
    volatile int* const __restrict__, const float, const size_t);

template __global__ void cuSZp_decompress_kernel_1D_outlier_f32_block_size_N<32>(
    float* const __restrict__, const unsigned char* const __restrict__,
    volatile unsigned int* const __restrict__, volatile unsigned int* const __restrict__,
    volatile int* const __restrict__, const float, const size_t);
template __global__ void cuSZp_decompress_kernel_1D_outlier_f32_block_size_N<64>(
    float* const __restrict__, const unsigned char* const __restrict__,
    volatile unsigned int* const __restrict__, volatile unsigned int* const __restrict__,
    volatile int* const __restrict__, const float, const size_t);
template __global__ void cuSZp_decompress_kernel_1D_outlier_f32_block_size_N<128>(
    float* const __restrict__, const unsigned char* const __restrict__,
    volatile unsigned int* const __restrict__, volatile unsigned int* const __restrict__,
    volatile int* const __restrict__, const float, const size_t);
template __global__ void cuSZp_decompress_kernel_1D_outlier_f32_block_size_N<256>(
    float* const __restrict__, const unsigned char* const __restrict__,
    volatile unsigned int* const __restrict__, volatile unsigned int* const __restrict__,
    volatile int* const __restrict__, const float, const size_t);
