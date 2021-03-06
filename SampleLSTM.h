#pragma once

#include <RTNeural/RTNeural.h>

/**
 * Modification of RTNeural::LSTMLayerT to use a variable recurrent delay
 * instead of the standard 1-sample recurrent delay.
 */
template <typename T, int in_sizet, int out_sizet>
class SampleLSTM
{
    using v_type = xsimd::simd_type<T>;
    static constexpr auto v_size = (int)v_type::size;
    static constexpr auto v_in_size = RTNeural::ceil_div(in_sizet, v_size);
    static constexpr auto v_out_size = RTNeural::ceil_div(out_sizet, v_size);

public:
    static constexpr auto in_size = in_sizet;
    static constexpr auto out_size = out_sizet;

    SampleLSTM();

    /** Returns the name of this layer. */
    std::string getName() const noexcept { return "lstm"; }

    /** Returns false since LSTM is not an activation. */
    constexpr bool isActivation() const noexcept { return false; }

    /** Prepares the LSTM to process samples with a given delay length. */
    void prepare (T delaySamples);

    /** Resets the state of the LSTM. */
    void reset();

    /** Performs forward propagation for this layer. */
    void forward(const v_type (&ins)[v_in_size])
    {
        // compute ft
        recurrent_mat_mul(outs, Uf, ft);
        for(int i = 0; i < v_out_size; ++i)
            ft[i] = sigmoid(xsimd::fma(Wf_1[i], ins[0], ft[i] + bf[i]));

        // compute it
        recurrent_mat_mul(outs, Ui, it);
        for(int i = 0; i < v_out_size; ++i)
            it[i] = sigmoid(xsimd::fma(Wi_1[i], ins[0], it[i] + bi[i]));

        // compute ot
        recurrent_mat_mul(outs, Uo, ot);
        for(int i = 0; i < v_out_size; ++i)
            ot[i] = sigmoid(xsimd::fma(Wo_1[i], ins[0], ot[i] + bo[i]));

        // compute ct
        recurrent_mat_mul(outs, Uc, ht);
        for(int i = 0; i < v_out_size; ++i)
            ct_internal[delayWriteIdx][i] = xsimd::fma(it[i], xsimd::tanh(xsimd::fma(Wc_1[i], ins[0], ht[i] + bc[i])), ft[i] * ct[i]);

        // compute output
        for(int i = 0; i < v_out_size; ++i)
            outs_internal[delayWriteIdx][i] = ot[i] * xsimd::tanh(ct_internal[delayWriteIdx][i]);

        processDelay (ct_internal, ct, delayWriteIdx);
        processDelay (outs_internal, outs, delayWriteIdx);
    }

    /**
     * Sets the layer kernel weights.
     *
     * The weights vector must have size weights[in_size][4 * out_size]
     */
    void setWVals(const std::vector<std::vector<T>>& wVals);

    /**
     * Sets the layer recurrent weights.
     *
     * The weights vector must have size weights[out_size][4 * out_size]
     */
    void setUVals(const std::vector<std::vector<T>>& uVals);

    /**
     * Sets the layer bias.
     *
     * The bias vector must have size weights[4 * out_size]
     */
    void setBVals(const std::vector<T>& bVals);

    v_type outs[v_out_size];

private:
    void processDelay (std::vector<std::array<v_type, v_out_size>>& delayVec, v_type (&out)[v_out_size], int delayWriteIndex)
    {
        for (int i = 0; i < v_out_size; ++i)
            out[i] = delayPlus1Mult * delayVec[0][i] + delayMult * delayVec[1][i];

        for (int j = 0; j < delayWriteIndex; ++j)
        {
            for (int i = 0; i < v_out_size; ++i)
                delayVec[j][i] = delayVec[j + 1][i];
        }
    }

    static inline void recurrent_mat_mul(const v_type (&vec)[v_out_size], const v_type (&mat)[out_size][v_out_size], v_type (&out)[v_out_size]) noexcept
    {
        T sums alignas(RTNEURAL_DEFAULT_ALIGNMENT)[out_size] { (T)0 };
        for(int i = 0; i < v_size; ++i)
        {
            for(int j = 0; j < v_out_size; ++j)
            {
                for(int k = 0; k < v_out_size; ++k)
                    sums[i + j * v_size] += xsimd::hadd(mat[i + j * v_size][k] * vec[k]);
            }
        }

        for(int i = 0; i < v_out_size; ++i)
            out[i] = xsimd::load_aligned(sums + i * v_size);
    }

    static inline v_type sigmoid(v_type x) noexcept
    {
        return (T)1.0 / ((T)1.0 + xsimd::exp(-x));
    }

    // kernel weights
    v_type Wf[out_size][v_in_size];
    v_type Wi[out_size][v_in_size];
    v_type Wo[out_size][v_in_size];
    v_type Wc[out_size][v_in_size];
    v_type kernel_outs[v_out_size];

    // single-input kernel weights
    v_type Wf_1[v_out_size];
    v_type Wi_1[v_out_size];
    v_type Wo_1[v_out_size];
    v_type Wc_1[v_out_size];

    // recurrent weights
    v_type Uf[out_size][v_out_size];
    v_type Ui[out_size][v_out_size];
    v_type Uo[out_size][v_out_size];
    v_type Uc[out_size][v_out_size];

    // biases
    v_type bf[v_out_size];
    v_type bi[v_out_size];
    v_type bo[v_out_size];
    v_type bc[v_out_size];

    // intermediate vars
    v_type ft[v_out_size];
    v_type it[v_out_size];
    v_type ot[v_out_size];
    v_type ht[v_out_size];
    v_type ct[v_out_size];

    std::vector<std::array<v_type, v_out_size>> ct_internal;
    std::vector<std::array<v_type, v_out_size>> outs_internal;
    int delayWriteIdx = 0;
    v_type delayMult = (T) 1;
    v_type delayPlus1Mult = (T) 0;
};

#include "SampleLSTM.tpp"
