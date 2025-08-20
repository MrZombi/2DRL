#include <string>
#include "core/Environment.h"
#include "render/Renderer.h"
#include "raylib.h"

void run_server(Environment& env);

static bool hasFlag(int argc, char** argv, const char* flag) {
    for (int i = 1; i < argc; ++i) if (std::string(argv[i]) == flag) return true;
    return false;
}
static std::string getOpt(int argc, char** argv, const char* key, const std::string& def="") {
    const std::string k(key);
    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        if (a.rfind(k, 0) == 0) {
            auto pos = a.find('=');
            if (pos != std::string::npos) return a.substr(pos+1);
        }
    }
    return def;
}

int main(int argc, char** argv) {
    // Defaults
    bool serverMode   = hasFlag(argc, argv, "--server");
    bool headless     = hasFlag(argc, argv, "--headless");
    int  renderEvery  = std::max(1, std::atoi(getOpt(argc, argv, "--render-every=", "1").c_str()));
    int  frameSkipCli = std::max(1, std::atoi(getOpt(argc, argv, "--frame-skip=", "1").c_str()));
    uint64_t seed     = std::strtoull(getOpt(argc, argv, "--seed=", "42").c_str(), nullptr, 10);
    std::string agent = getOpt(argc, argv, "--agent=", "human");
    std::string runId = getOpt(argc, argv, "--run_id=", "cli");

    Environment env;
    Renderer    renderer;

    if (serverMode) {
        env.setMode(Mode::Agent);
        env.setRenderEnabled(!headless);
        env.setRenderEvery(renderEvery);
        env.setFrameSkip(frameSkipCli);
        env.setRunInfo(agent, runId);

        if (!headless) {
            env.setRenderer(&renderer);
        } else {
            env.setRenderer(nullptr);
        }
        run_server(env);
        return 0;
    }

    env.setMode(Mode::Player);
    env.setRenderEnabled(true);
    env.setRenderEvery(1);
    env.setFrameSkip(1);
    env.setRunInfo("human", "local");
    env.setRenderer(&renderer);

    renderer.initIfNeeded();
    env.reset(seed);

    while (!WindowShouldClose()) {
        auto res = env.step(1);
        if (res.done) {
            seed += 1;
            env.reset(seed);
        }
    }

    renderer.shutdown();
    return 0;
}