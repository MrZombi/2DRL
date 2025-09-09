#pragma once
#include <string>

class EpisodeLogger {
public:
    explicit EpisodeLogger(const std::string& filename);

    void logEpisode(
        const std::string& run_id,
        const std::string& agent,
        int episode,
        unsigned long long seed,
        float ep_return,
        int score,
        int steps,
        int pellets,
        int power_pellets,
        int ghosts,
        int deaths,
        bool cleared,
        bool timeout,
        int frameskip,
        bool render,
        long long duration_ms
    );

private:
    std::string filename;
    void ensureHeader();
    bool headerWritten = false;
};