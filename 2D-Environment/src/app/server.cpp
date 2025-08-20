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

#include "third_party/cpp-httplib/httplib.h"
#include "Environment.h"
#include "Action.h"

#include <regex>
#include <sstream>
#include <iomanip>
#include <iostream>

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

static Action parse_action(const std::string& body) {
    std::string s; json_string(body, "action", "Stay", s);
    if      (s == "Up")    return Action::Up;
    else if (s == "Right") return Action::Right;
    else if (s == "Down")  return Action::Down;
    else if (s == "Left")  return Action::Left;
    return Action::Stay;
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

    svr.Get("/spec", [](const Request&, Response& res){
        std::ostringstream oss;
        oss << R"({"action_space":{"n":5,"actions":["Up","Right","Down","Left","Stay"]},)"
               R"("observation":{"type":"vector","size":64},)"
               R"("supports":{"render":true,"frame_skip":true,"seed":true,"render_every":true}})";
        res.set_content(oss.str(), "application/json");
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
        oss << R"({"obs":)" << obs_to_json(obs) << R"(,"info":{}})";
        res.set_content(oss.str(), "application/json");
    });

    svr.Post("/set_mode", [&](const Request& req, Response& res){
        bool has = false; bool renderVal=false; int fs=0; int re=0;

        if (json_bool(req.body, "render", false, renderVal)) env.setRenderEnabled(renderVal);
        if (json_int<int>(req.body, "frame_skip", 0, fs) && fs>0) env.setFrameSkip(fs);
        if (json_int<int>(req.body, "render_every", 0, re) && re>0) env.setRenderEvery(re);

        res.set_content(R"({"ok":true})", "application/json");
    });

    svr.Post("/step", [&](const Request& req, Response& res){
        Action a = parse_action(req.body);
        env.setPendingAction(a);

        int fs_override = 0;
        if (json_int<int>(req.body, "frame_skip", 0, fs_override) && fs_override > 0) {
            auto r = env.step(fs_override);
            std::ostringstream oss;
            oss << R"({"obs":)" << obs_to_json(r.obs)
                << R"(,"reward":)" << std::fixed << std::setprecision(6) << r.reward
                << R"(,"done":)" << (r.done ? "true" : "false")
                << R"(,"info":{}})";
            res.set_content(oss.str(), "application/json");
            return;
        }

        auto r = env.step(1);
        std::ostringstream oss;
        oss << R"({"obs":)" << obs_to_json(r.obs)
            << R"(,"reward":)" << std::fixed << std::setprecision(6) << r.reward
            << R"(,"done":)" << (r.done ? "true" : "false")
            << R"(,"info":{}})";
        res.set_content(oss.str(), "application/json");
    });

    const char* host = "127.0.0.1";
    int port = 8000;
    std::cout << "Server listening on http://" << host << ":" << port << std::endl;
    svr.listen(host, port);
}
