#include "Environment.h"
// Entfernt:
// #include "raylib.h"
// #include "Renderer.h"
#include "Constants.h"

#include <algorithm>
#include <random>
#include <filesystem>
#include <cctype>

Environment::Environment() : pacman_(Constants::Cols/2, Constants::Rows/2) {
    static const int LEVEL[Constants::Rows][Constants::Cols] = {
        {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},
        {1,2,2,2,1,2,2,2,2,2,2,2,2,2,2,2,2,2,1,2,2,2,1},
        {1,2,1,2,1,2,1,1,1,1,1,2,1,1,1,1,1,2,1,2,1,2,1},
        {1,2,2,2,2,2,2,2,1,2,2,2,2,2,1,2,2,2,2,2,2,2,1},
        {1,1,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,1,1},
        {1,2,2,2,1,2,1,2,1,2,2,2,2,2,1,2,1,2,1,2,2,2,1},
        {1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1},
        {1,2,2,2,1,2,1,2,2,2,2,2,2,2,2,2,2,2,1,2,2,2,1},
        {1,1,1,1,1,2,1,2,1,1,1,1,1,1,1,2,1,2,1,1,1,1,1},
        {1,2,2,2,2,2,1,2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,1},
        {1,2,1,1,1,1,1,2,1,1,1,3,1,1,1,2,1,1,1,1,1,2,1},
        {2,2,2,2,2,2,2,2,1,0,0,0,0,0,1,2,2,2,2,2,2,2,2},
        {1,2,1,1,1,1,1,2,1,1,1,1,1,1,1,2,1,1,1,1,1,2,1},
        {1,2,2,2,2,2,1,2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,1},
        {1,1,1,1,1,2,1,2,1,1,1,1,1,1,1,2,1,2,1,1,1,1,1},
        {1,2,2,2,1,2,1,2,2,2,2,2,2,2,2,2,1,2,1,2,2,2,1},
        {1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1},
        {1,2,2,2,1,2,1,2,1,2,2,2,2,2,1,2,1,2,1,2,2,2,1},
        {1,1,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,1,1},
        {1,2,2,2,2,2,2,2,1,2,2,2,2,2,1,2,2,2,2,2,2,2,1},
        {1,2,1,2,1,2,1,1,1,1,1,2,1,1,1,1,1,2,1,2,1,2,1},
        {1,2,2,2,1,2,2,2,2,2,2,2,2,2,2,2,2,2,1,2,2,2,1},
        {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},
    };

    originalMaze_.assign(Constants::Rows, std::vector<int>(Constants::Cols, 0));
    for (int y = 0; y < Constants::Rows; ++y) {
        for (int x = 0; x < Constants::Cols; ++x) {
            originalMaze_[y][x] = LEVEL[y][x];
        }
    }
    maze_ = originalMaze_;
}

void Environment::setRunInfo(const std::string& agent, const std::string& runId) {
    agentName_ = sanitizeAgent_(agent.empty() ? "unknown" : agent);
    runId_     = runId.empty() ? "local-run" : runId;

    if (agentName_ == "sb3" || agentName_ == "SmokeTest") {
        logger_.reset();
        return;
    }

    const std::string path = buildLogPath_(agentName_, runId_);
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    logger_ = std::make_unique<EpisodeLogger>(path);
}

std::vector<float> Environment::reset(uint64_t seed) {
    rng_.seed(seed);
    stepCounter_      = 0;
    pelletsCollected_ = 0;
    powerCollected_   = 0;
    ghostsEaten_      = 0;
    deathsThisEp_     = 0;
    clearedLastStep_  = false;
    diedLastStep_     = false;
    powerTimer_       = 0;
    gameOver_         = false;
    scoreLastStep_    = 0;
    pendingAction_    = Action::Stay;
    episodeStartTime_ = std::chrono::steady_clock::now();
    currentSeed_ = seed;
    ++currentEpisode_;
    episodeReturn_ = 0.0f;

    resetMaze_();

    {
        auto [px, py] = randomFreeCell_();
        pacman_.x = px; pacman_.y = py;
        pacman_.dirX = 0; pacman_.dirY = 0;
        pacman_.score = 0;
    }

    {
        ghosts_.clear();
        constexpr int cx = Constants::Cols / 2;
        constexpr int cy = Constants::Rows / 2;

        ghosts_.emplace_back(cx    , cy    , GhostType::BLINKY);
        ghosts_.emplace_back(cx - 1, cy    , GhostType::PINKY);
        ghosts_.emplace_back(cx + 1, cy    , GhostType::INKY);
        ghosts_.emplace_back(cx + 2, cy    , GhostType::CLYDE);

        for (auto& g : ghosts_) g.frightened = false;
    }

    pill_  = Pickup{};
    fruit_ = Pickup{};

    {
        int count = 0;
        const int H = static_cast<int>(maze_.size());
        const int W = H ? static_cast<int>(maze_[0].size()) : 0;
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                if (maze_[y][x] == static_cast<int>(Constants::CellType::Coin)) ++count;
            }
        }
        pelletsAtStart_ = count;
    }
    return makeObservation_();
}

StepResult Environment::step(int n_ticks) {
    StepResult out;
    out.reward = 0.0f;
    out.done   = false;

    for (int i = 0; i < n_ticks && !out.done; ++i) {
        ++stepCounter_;

        clearedLastStep_ = false;
        diedLastStep_    = false;

        Action a = getAction_();

        pacman_.applyAction(a, maze_);

        if (pacman_.y >= 0 && pacman_.y < static_cast<int>(maze_.size()) && pacman_.x >= 0 && pacman_.x < static_cast<int>(maze_[0].size()))
        {
            if (maze_[pacman_.y][pacman_.x] == static_cast<int>(Constants::CellType::Coin)) {
                maze_[pacman_.y][pacman_.x] = static_cast<int>(Constants::CellType::Empty);
                pacman_.score += 10;
                ++pelletsCollected_;
            }
        }

        const bool powerWasZero = (powerTimer_ == 0);
        updateItemsAndPower_();
        if (powerWasZero && powerTimer_ > 0) {
            for (auto& g : ghosts_) {
                g.setFrightened(Constants::PowerDuration);
            }
        }

        for (auto& g : ghosts_) {
            g.updateState();
            g.move(maze_, ghosts_, pacman_.x, pacman_.y, rng_);
        }

        for (auto& g : ghosts_) {
            if (g.x == pacman_.x && g.y == pacman_.y) {
                if (g.frightened) {
                    pacman_.score += 200;
                    ++ghostsEaten_;
                    g.frightened = false;
                    g.x = Constants::Cols / 2;
                    g.y = Constants::Rows / 2;
                } else {
                    gameOver_     = true;
                    diedLastStep_ = true;
                    ++deathsThisEp_;
                }
            }
        }

        if (pelletsAtStart_ > 0 && pelletsCollected_ >= pelletsAtStart_) {
            gameOver_ = true;
            clearedLastStep_ = true;
        }

        {
            const int   scoreDelta = pacman_.score - scoreLastStep_;
            float r = static_cast<float>(scoreDelta) - 0.01f;
            if (clearedLastStep_) r += 200.0f;
            if (diedLastStep_)    r -= 500.0f;
            out.reward += r;
            scoreLastStep_ = pacman_.score;
        }

        if (gameOver_ || stepCounter_ >= timeoutSteps_) {
            out.done = true;
        }
    }

    out.obs = makeObservation_();

    if (out.done && logger_) {
        const int pelletsLeft = pelletsAtStart_ - pelletsCollected_;
        const bool cleared = (pelletsAtStart_ > 0 && pelletsLeft == 0);
        const bool timeout = (!cleared && deathsThisEp_ == 0 && stepCounter_ >= timeoutSteps_);

        auto end = std::chrono::steady_clock::now();
        long long durationMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - episodeStartTime_).count();

        logger_->logEpisode(
            runId_, agentName_,
            currentEpisode_, static_cast<unsigned long long>(currentSeed_),
            static_cast<float>(pacman_.score),
            pacman_.score, stepCounter_,
            pelletsCollected_, powerCollected_,
            ghostsEaten_, deathsThisEp_,
            cleared, timeout,
            frameSkip_, renderEnabled_,
            durationMs
        );
    }
    return out;
}

std::vector<float> Environment::makeObservation_() const {
    constexpr int W = Constants::Cols;
    constexpr int H = Constants::Rows;

    auto clamp01 = [](float v){ return v < 0.f ? 0.f : (v > 1.f ? 1.f : v); };

    std::vector<float> obs;
    obs.reserve(64);

    obs.push_back(static_cast<float>(pacman_.x) / static_cast<float>(W - 1));
    obs.push_back(static_cast<float>(pacman_.y) / static_cast<float>(H - 1));

    int dir = 4;
    if (pacman_.dirX == 0 && pacman_.dirY == -1) dir = 0;
    else if (pacman_.dirX == 1 && pacman_.dirY == 0) dir = 1;
    else if (pacman_.dirX == 0 && pacman_.dirY == 1) dir = 2;
    else if (pacman_.dirX == -1 && pacman_.dirY == 0) dir = 3;
    for (int i = 0; i < 4; ++i) obs.push_back(i == dir ? 1.f : 0.f);

    auto passable = [&](int x, int y) {
        if (x < 0 || x >= W || y < 0 || y >= H) return false;
        int c = maze_[y][x];
        return c != static_cast<int>(Constants::CellType::Wall) &&
               c != static_cast<int>(Constants::CellType::Gate);
    };
    obs.push_back(passable(pacman_.x,     pacman_.y - 1) ? 1.f : 0.f); // up
    obs.push_back(passable(pacman_.x + 1, pacman_.y    ) ? 1.f : 0.f); // right
    obs.push_back(passable(pacman_.x,     pacman_.y + 1) ? 1.f : 0.f); // down
    obs.push_back(passable(pacman_.x - 1, pacman_.y    ) ? 1.f : 0.f); // left

    auto norm_rel = [&](int dx, int dy) {
        float fx = static_cast<float>(dx) / static_cast<float>(W - 1);
        float fy = static_cast<float>(dy) / static_cast<float>(H - 1);
        return std::pair<float,float>{ clamp01(0.5f + fx * 0.5f),
                                       clamp01(0.5f + fy * 0.5f) };
    };
    for (const auto& g : ghosts_) {
        int dx = g.x - pacman_.x;
        int dy = g.y - pacman_.y;
        auto [nx, ny] = norm_rel(dx, dy);
        obs.push_back(nx);
        obs.push_back(ny);
        obs.push_back(g.frightened ? 1.f : 0.f);
    }

    int   best_dx = 0, best_dy = 0;
    float best_d  = 1e9f;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            if (maze_[y][x] == static_cast<int>(Constants::CellType::Coin)) {
                int   dx = x - pacman_.x;
                int   dy = y - pacman_.y;
                auto d  = static_cast<float>(std::abs(dx) + std::abs(dy));
                if (d < best_d) { best_d = d; best_dx = dx; best_dy = dy; }
            }
        }
    }
    if (best_d < 1e8f) {
        auto [nx, ny] = norm_rel(best_dx, best_dy);
        obs.push_back(nx);
        obs.push_back(ny);
        auto maxd = static_cast<float>(W + H);
        obs.push_back(clamp01(best_d / maxd));
    } else {
        obs.push_back(0.f); obs.push_back(0.f); obs.push_back(1.f);
    }

    int pelletsLeft = pelletsAtStart_ - pelletsCollected_;
    float pellets_left_norm = (pelletsAtStart_ > 0)
        ? static_cast<float>(pelletsLeft) / static_cast<float>(pelletsAtStart_)
        : 0.f;
    obs.push_back(clamp01(pellets_left_norm));
    obs.push_back(clamp01(static_cast<float>(stepCounter_) /
                          static_cast<float>(std::max(1, timeoutSteps_))));

    int aidx = static_cast<int>(pendingAction_);
    for (int i = 0; i < 5; ++i) obs.push_back(i == aidx ? 1.f : 0.f);

    while (obs.size() < 64) obs.push_back(0.0f);

    return obs;
}

std::pair<int,int> Environment::randomFreeCell_() {
    std::uniform_int_distribution<int> distX(1, Constants::Cols - 2);
    std::uniform_int_distribution<int> distY(1, Constants::Rows - 2);

    for (int tries = 0; tries < 10000; ++tries) {
        int x = distX(rng_);
        int y = distY(rng_);
        int c = maze_[y][x];
        if (c != static_cast<int>(Constants::CellType::Wall) &&
            c != static_cast<int>(Constants::CellType::Gate)) {
            return {x, y};
        }
    }
    return {1,1};
}

void Environment::resetMaze_() {
    maze_ = originalMaze_;
}

void Environment::updateItemsAndPower_()
{
    if (pill_.active && pacman_.x == pill_.x && pacman_.y == pill_.y)
    {
        if (pill_.replacedCoin == static_cast<int>(Constants::CellType::Coin)) {
            maze_[pill_.y][pill_.x] = static_cast<int>(Constants::CellType::Coin);
        }
        pill_.x = pill_.y = -1;
        pill_.active = false;
        pill_.respawnTimer = Constants::PowerRespawnTime;
        powerTimer_       = Constants::PowerDuration;
        ++powerCollected_;
    }

    if (!pill_.active)
    {
        if (pill_.respawnTimer > 0) {
            --pill_.respawnTimer;
        } else {
            std::uniform_int_distribution<int> distX(1, Constants::Cols - 2);
            std::uniform_int_distribution<int> distY(1, Constants::Rows - 2);
            int rx = distX(rng_);
            int ry = distY(rng_);

            if (maze_[ry][rx] != static_cast<int>(Constants::CellType::Wall))
            {
                pill_.x            = rx;
                pill_.y            = ry;
                pill_.type         = 1;
                pill_.active       = true;
                pill_.respawnTimer = Constants::PickupActiveTime;
                pill_.replacedCoin = (maze_[ry][rx] == static_cast<int>(Constants::CellType::Coin))
                    ? static_cast<int>(Constants::CellType::Coin) : 0;
                maze_[ry][rx]      = 0;
            }
        }
    }

    if (fruit_.active && pacman_.x == fruit_.x && pacman_.y == fruit_.y)
    {
        if (fruit_.replacedCoin == static_cast<int>(Constants::CellType::Coin)) {
            maze_[fruit_.y][fruit_.x] = static_cast<int>(Constants::CellType::Coin);
        }
        fruit_.x = -1;
        fruit_.y = -1;
        fruit_.active       = false;
        fruit_.respawnTimer = Constants::FruitRespawnTime;
        pacman_.score      += 100;
    }

    if (!fruit_.active)
    {
        if (fruit_.respawnTimer > 0) {
            --fruit_.respawnTimer;
        } else {
            std::uniform_int_distribution<int> distX(1, Constants::Cols - 2);
            std::uniform_int_distribution<int> distY(1, Constants::Rows - 2);
            int rx = distX(rng_);
            int ry = distY(rng_);

            if (maze_[ry][rx] != static_cast<int>(Constants::CellType::Wall))
            {
                fruit_.x            = rx;
                fruit_.y            = ry;
                fruit_.type         = 2;
                fruit_.active       = true;
                fruit_.respawnTimer = Constants::PickupActiveTime;
                fruit_.replacedCoin = (maze_[ry][rx] == static_cast<int>(Constants::CellType::Coin))
                    ? static_cast<int>(Constants::CellType::Coin) : 0;
                maze_[ry][rx]       = 0;
            }
        }
    }
    if (powerTimer_ > 0) {
        --powerTimer_;
    }
}

Action Environment::getAction_() const {
    return pendingAction_;
}

std::string Environment::sanitizeAgent_(std::string s) {
    for (auto& ch : s) {
        if (!std::isalnum(static_cast<unsigned char>(ch)) && ch!='_' && ch!='-') ch = '_';
    }
    if (s.empty()) s = "unknown";
    return s;
}

std::string Environment::buildLogPath_(const std::string& agent, const std::string& runId) {
    return "logs/" + agent + "/" + agent + "__run-" + runId + ".csv";
}