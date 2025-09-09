#include "EpisodeLogger.h"
#include <fstream>
#include <filesystem>

EpisodeLogger::EpisodeLogger(const std::string& filename)
    : filename(filename) {}

void EpisodeLogger::ensureHeader() {
    if (headerWritten) return;
    if (!std::filesystem::exists(filename)) {
            std::ofstream f(filename);
            f << "run_id,agent,episode,seed,return,score,steps,pellets,power_pellets,ghosts,deaths,cleared,timeout,"
                 "frameskip,render,duration_ms\n";
    }
    headerWritten = true;
}

void EpisodeLogger::logEpisode(
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
) {
    ensureHeader();
    std::ofstream f(filename, std::ios::app);
    f << run_id << "," << agent << "," << episode << "," << seed << ","
      << ep_return << "," << score << "," << steps << ","
      << pellets << "," << power_pellets << "," << ghosts << "," << deaths << ","
      << (cleared ? "True" : "False") << "," << (timeout ? "True" : "False") << ","
      << frameskip << "," << (render ? "True" : "False") << "," << duration_ms << "\n";
}
