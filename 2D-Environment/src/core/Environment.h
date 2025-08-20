#pragma once
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <filesystem>
#include <regex>
#include <memory>
#include "Action.h"
#include "EpisodeLogger.h"
#include "Pacman.h"
#include "Ghost.h"

enum class Mode { Player, Agent };

struct StepResult {
    std::vector<float> obs;
    float reward = 0.0f;
    bool  done   = false;
};

class Renderer;

class Environment {
public:
    Environment();

    std::vector<float> reset(uint64_t seed);

    StepResult step(int n_ticks);

    void setMode(Mode m)              { mode_ = m; }
    void setRenderEnabled(bool b)     { renderEnabled_ = b; }
    void setRenderEvery(int n)        { renderEvery_ = (n <= 0 ? 1 : n); }
    void setPendingAction(Action a)   { pendingAction_ = a; }
    void setFrameSkip(int k) { frameSkip_ = (k <= 0 ? 1 : k); }

    void setRenderer(Renderer* r) {renderer_ =r; }

    void setRunInfo(const std::string& agent, const std::string& runId);

    const std::vector<std::vector<int>>& maze()    const { return maze_; }
    const std::vector<Ghost>& ghosts()             const { return ghosts_; }
    const Pacman& pacman()                         const { return pacman_; }
    int  powerTimer()                              const { return powerTimer_; }
    bool renderEnabled()                           const { return renderEnabled_; }
    int renderEvery()                             const { return renderEvery_; }

private:
    std::vector<float> makeObservation_() const;
    void resetMaze_();
    void updateItemsAndPower_();
    void maybeRender_();
    Action getAction_() const;
    std::pair<int, int> randomFreeCell_();
    static std::string sanitizeAgent_(std::string s);
    static std::string buildLogPath_(const std::string& agent, const std::string& runId);
    Renderer* renderer_ = nullptr;

    std::vector<std::vector<int>> originalMaze_;
    std::vector<std::vector<int>> maze_;
    Pacman pacman_;
    std::vector<Ghost> ghosts_;
    int powerTimer_ = 0;

    int stepCounter_      = 0;
    int timeoutSteps_     = 4000;
    int pelletsAtStart_   = 0;
    int pelletsCollected_ = 0;
    int powerCollected_   = 0;
    int ghostsEaten_      = 0;
    int deathsThisEp_     = 0;

    bool gameOver_ = false;
    int scoreLastStep_ = 0;

    Action pendingAction_ = Action::Stay;

    bool clearedLastStep_ = false;
    bool diedLastStep_    = false;

    Mode   mode_          = Mode::Player;
    bool   renderEnabled_ = true;
    int    renderEvery_   = 1;
    int    renderCounter_ = 0;

    std::string agentName_;
    std::string runId_;
    std::chrono::steady_clock::time_point episodeStartTime_;
    std::unique_ptr<EpisodeLogger> logger_;
    uint64_t currentSeed_   = 0;
    int      currentEpisode_ = 0;
    float    episodeReturn_  = 0.0f;
    int      frameSkip_      = 1;

    struct Pickup {
        int  x = -1;
        int  y = -1;
        int  type = 0;
        bool active = false;
        int  respawnTimer = 0;
        int  replacedCoin = 0;
    };

    Pickup pill_;
    Pickup fruit_;

    std::mt19937_64 rng_{0};
};
