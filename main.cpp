#include <iostream>
#include <matplotlibcpp.h>

#include "ResampledRNN.h"

namespace plt = matplotlibcpp;
using RNNType = ResampledRNN<28>;
constexpr auto trainingSampleRate = 96000.0f;

auto genSine(float freq, float numSeconds, float sampleRate)
{
    const auto nSamples = int(numSeconds * sampleRate);
    std::vector<float> time(nSamples);
    std::iota(time.begin(), time.end(), 0.0f);
    std::transform(time.begin(), time.end(), time.begin(), [Ts = 1.0f / sampleRate] (auto x) { return x * Ts; });

    std::vector<float> sine(nSamples);
    std::transform(time.begin(), time.end(), sine.begin(), [freq] (auto t) { return std::sin(2.0f * (float) M_PI * freq * t); });

    return std::make_pair(std::move(time), std::move(sine));
}

void processRNN(RNNType& rnn, float sampleRate, bool corrected = true, const std::map<std::string, std::string>& keywords = {}, bool plotInput = false)
{
    constexpr auto freq = 100.0f;
    constexpr auto seconds = 0.1f;
    auto [time, sine] = genSine(freq, seconds, sampleRate);

    if (plotInput)
        plt::plot(time, sine);

    rnn.prepare (corrected ? sampleRate : trainingSampleRate);
    rnn.process(sine.data(), (int) sine.size());
    plt::plot(time, sine, keywords);
}

void testSampleRateRatio(float ratio, bool corrected = true)
{
    plt::figure();

    std::ostringstream ratioStr;
    ratioStr.precision(2);
    ratioStr << std::fixed << ratio;
    std::string title = "RNN Processing " + ratioStr.str() + "x Sample Rate";
    if(corrected)
        title += " (Corrected)";
    plt::title(title);
    plt::grid(true);

    RNNType rnn;
    rnn.initialise("test_model.json", trainingSampleRate);
    processRNN(rnn, trainingSampleRate, true, {{"label", "Training Sample Rate"}});
    processRNN(rnn, ratio * trainingSampleRate, corrected, {{"label", ratioStr.str() + "x Training Sample Rate"}, {"linestyle", "--"}});

    plt::legend();
    plt::xlim(0.01f, 0.05f);
    plt::xlabel("Time [seconds]");
    plt::ylabel("Amplitude");
    plt::save("figures/" + title + ".png");
}

int main()
{
    testSampleRateRatio(2.0f, false);
    testSampleRateRatio(2.0f);
    testSampleRateRatio(3.0f);
    testSampleRateRatio(1.5f);
    testSampleRateRatio(1.25f);
    testSampleRateRatio(1.75f);
    testSampleRateRatio(4.33f);

    plt::show();

    return 0;
}
