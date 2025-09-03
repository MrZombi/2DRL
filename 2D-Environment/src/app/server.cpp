#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef NOGDI
#define NOGDI
#endif
#ifndef NOUSER
#define NOUSER
#endif
#endif

#include "cpp-httplib/httplib.h"
#include "Environment.h"
#include "Action.h"

// 👉 Pfad ggf. anpassen (von src/app/ nach src/third_party/)
#include "../third_party/json/json.hpp"
#include <regex>
#include <sstream>
#include <iomanip>
#include <iostream>

using json = nlohmann::json;

static bool json_bool(const std::string& body, const char* key, bool def, bool& out) {
    std::regex rgx(std::string("\"") + key + R"("\s*:\s*(true|false))", std::regex::icase);
    std::smatch m; if (std::regex_search(body, m, rgx)) { out = (m[1] == "true" || m[1] == "TRUE"); return true; }
    out = def; return false;
}

template<typename T>
static bool json_int(const std::string& body, const char* key, T def, T& out) {
    std::regex rgx(std::string("\"") + key + R"("\s*:\s*(-?\d+))");
    std::smatch m; if (std::regex_search(body, m, rgx)) { long long v = std::stoll(m[1]); out = static_cast<T>(v); return true; }
    out = def; return false;
}

static bool json_string(const std::string& body, const char* key, const std::string& def, std::string& out) {
    std::regex rgx(std::string("\"") + key + R"_("\s*:\s*"(.*?)")_");
    std::smatch m; if (std::regex_search(body, m, rgx)) { out = m[1].str(); return true; }
    out = def; return false;
}

static Action parse_action_label(const std::string& s) {
    if      (s == "Up")    return Action::Up;
    else if (s == "Right") return Action::Right;
    else if (s == "Down")  return Action::Down;
    else if (s == "Left")  return Action::Left;
    return Action::Stay;
}

static inline std::string action_label_from_index(int idx) {
    static const char* kLabels[5] = {"Up","Right","Down","Left","Stay"};
    if (idx < 0 || idx >= 5) return {};
    return kLabels[idx];
}

static std::string obs_to_json(const std::vector<float>& v) {
    std::ostringstream oss;
    oss << '[';
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) oss << ',';
        oss << std::fixed << std::setprecision(6) << v[i];
    }
    oss << ']';
    return oss.str();
}

static std::string info_to_json(const Environment& env) {
    std::ostringstream oss;
    oss << R"({"player_cell":[)" << env.pacmanX() << "," << env.pacmanY() << "]"
        << R"(,"pellets_left":)"   << (env.pelletsAtStart() - env.pelletsCollected())
        << R"(,"died":)"           << (env.diedLastStep()    ? "true" : "false")
        << R"(,"cleared":)"        << (env.clearedLastStep() ? "true" : "false")
        << R"(,"power_timer":)"    << env.powerTimer()
        << R"(,"ghosts_eaten":)"   << env.ghostsEaten()
        << R"(,"power_collected":)"<< env.powerCollected()
        << R"(,"step":)"           << env.stepCounter()
        << "}";
    return oss.str();
}

void run_server(Environment& env) {
    using namespace httplib;
    Server svr;

    svr.set_default_headers({
        {"Access-Control-Allow-Origin", "*"},
        {"Access-Control-Allow-Headers", "Content-Type"},
        {"Access-Control-Allow-Methods", "GET,POST,OPTIONS"}
    });
    svr.Options(R"((.*))", [](const Request&, Response& res){ res.status = 200; });

    svr.Get("/health", [](const Request&, Response& res){
        res.set_content(R"({"status":"ok"})", "application/json");
    });

    svr.Get("/spec", [](const Request&, Response& res) {
        json spec = {
            {"action_space", {{"type","discrete"},{"n",5},{"labels",{"Up","Right","Down","Left","Stay"}}}},
            {"observation_space", {{"type","box"},{"size",64},{"dtype","float32"}}},
            {"info_keys", {"player_cell","pellets_left","died","cleared","timeout","terminated_reason",
                           "power_timer","ghosts_eaten","power_collected","step"}},
            {"supports", {{"render",true},{"frame_skip",true},{"seed",true},{"render_every",true}}}
        };
        res.set_content(spec.dump(), "application/json");
    });

    svr.Post("/reset", [&](const Request& req, Response& res){
        uint64_t seed = 42; json_int<uint64_t>(req.body, "seed", seed, seed);
        bool render = false; json_bool(req.body, "render", render, render);
        int frame_skip = 1;  json_int<int>(req.body, "frame_skip", frame_skip, frame_skip);
        int render_every = 1; json_int<int>(req.body, "render_every", render_every, render_every);
        std::string agent = "agent"; json_string(req.body, "agent", agent, agent);
        std::string run_id = "server"; json_string(req.body, "run_id", run_id, run_id);

        env.setMode(Mode::Agent);
        env.setRenderEnabled(render);
        env.setRenderEvery(render_every);
        env.setFrameSkip(frame_skip);
        env.setRunInfo(agent, run_id);

        const auto obs = env.reset(seed);
        std::ostringstream oss;
        oss << R"({"obs":)" << obs_to_json(obs)
            << R"(,"info":)" << info_to_json(env) << "}";
        res.set_content(oss.str(), "application/json");
    });

    svr.Post("/set_mode", [&](const Request& req, Response& res){
        bool renderVal = false;
        int fs = 0;
        int re = 0;

        if (json_bool(req.body, "render", false, renderVal)) {
            env.setRenderEnabled(renderVal);
        }
        if (json_int<int>(req.body, "frame_skip", 0, fs) && fs > 0) {
            env.setFrameSkip(fs);
        }
        if (json_int<int>(req.body, "render_every", 0, re) && re > 0) {
            env.setRenderEvery(re);
        }

        res.set_content(R"({"ok":true})", "application/json");
    });

    svr.Post("/step", [&](const Request& req, Response& res){
        std::string action_label;
        try {
            json j = json::parse(req.body, nullptr, true, true);
            if (!j.contains("action")) {
                res.status = 400;
                res.set_content(R"({"error":"missing 'action'"})", "application/json");
                return;
            }
            if (j["action"].is_number_integer()) {
                const int a = j["action"].get<int>();
                action_label = action_label_from_index(a);
                if (action_label.empty()) {
                    res.status = 400;
                    res.set_content("{\"error\":\"action index out of range (0..4)\"}", "application/json");
                    return;
                }
            } else if (j["action"].is_string()) {
                action_label = j["action"].get<std::string>();
            } else {
                res.status = 400;
                res.set_content("{\"error\":\"invalid 'action' type (int or string)\"}", "application/json");
                return;
            }
        } catch (const std::exception& e) {
            json err = {{"error","invalid json body"},{"what",e.what()}};
            res.status = 400;
            res.set_content(err.dump(), "application/json");
            return;
        }

        const Action act = parse_action_label(action_label);
        auto r = env.step(static_cast<int>(act));

        json info;
        try {
            info = json::parse(info_to_json(env));  // WICHTIG: mit (env)
        } catch (...) {
            info = json::object();
        }

        if (r.done) {
            const bool died    = info.value("died", false);
            const bool cleared = info.value("cleared", false);
            const bool timeout = (!died && !cleared);
            info["timeout"] = timeout;
            info["terminated_reason"] = died ? "died" : (cleared ? "cleared" : "timeout");
        } else {
            info["timeout"] = false;
            if (!info.contains("terminated_reason")) info["terminated_reason"] = "";
        }

        std::ostringstream oss;
        oss << "{\"obs\":"   << obs_to_json(r.obs)
            << ",\"reward\":"<< std::fixed << std::setprecision(6) << r.reward
            << ",\"done\":"  << (r.done ? "true" : "false")
            << ",\"info\":"  << info.dump()
            << "}";
        res.set_content(oss.str(), "application/json");
    });

    const char* host = "127.0.0.1";
    int port = 8000;
    std::cout << "Server listening on http://" << host << ":" << port << std::endl;
    svr.listen(host, port);
}
