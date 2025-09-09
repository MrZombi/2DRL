#include <string>
#include <vector>
#include <thread>
#include <memory>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include "Environment.h"

// deklariert in server.cpp
void run_server(Environment& env, const char* host, int port);

// ---------- kleine Hilfsfunktionen ----------

static bool hasFlag(int argc, char** argv, const char* flag) {
    for (int i = 1; i < argc; ++i) if (std::string(argv[i]) == flag) return true;
    return false;
}

static std::string getOpt(int argc, char** argv, const char* key, const std::string& def = "") {
    const std::string k(key);
    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        if (a.rfind(k, 0) == 0) {
            auto pos = a.find('=');
            if (pos != std::string::npos) return a.substr(pos + 1);
        }
    }
    return def;
}

static int parse_int(const std::string& s, int def) {
    if (s.empty()) return def;
    errno = 0;
    char* end = nullptr;
    long v = std::strtol(s.c_str(), &end, 10);
    if (errno != 0 || end == s.c_str() || *end != '\0') return def;
    if (v < std::numeric_limits<int>::min()) return std::numeric_limits<int>::min();
    if (v > std::numeric_limits<int>::max()) return std::numeric_limits<int>::max();
    return static_cast<int>(v);
}

static uint64_t parse_uint64(const std::string& s, uint64_t def) {
    if (s.empty()) return def;
    errno = 0;
    char* end = nullptr;
    unsigned long long v = std::strtoull(s.c_str(), &end, 10);
    if (errno != 0 || end == s.c_str() || *end != '\0') return def;
    return static_cast<uint64_t>(v);
}

static std::string trim(std::string s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.erase(s.begin());
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back())))  s.pop_back();
    return s;
}

// Ports-Parser: "8000-8015", "8000,8001,8002", "auto", "auto:16"
static std::vector<int> parse_ports(const std::string& portsArg,
                                    int fallback_single,
                                    int base_port_default = 8000) {
    std::vector<int> out;
    const std::string arg = trim(portsArg);

    if (!arg.empty()) {
        if (arg.rfind("auto", 0) == 0) {
            // "auto" oder "auto:N"
            int want = 0;
            auto colon = arg.find(':');
            if (colon != std::string::npos) {
                want = std::max(1, parse_int(arg.substr(colon + 1), 0));
            }
            unsigned int hc = std::thread::hardware_concurrency();
            if (hc == 0) hc = 4;
            if (want <= 0) {
                // konservativ: max 12, mind. 4, etwas Luft lassen
                unsigned int guess = (hc > 2) ? (hc - 2) : 4;
                if (guess < 4) guess = 4;
                if (guess > 12) guess = 12;
                want = static_cast<int>(guess);
            }
            const int base_port = base_port_default;
            for (int i = 0; i < want; ++i) out.push_back(base_port + i);
        } else if (arg.find('-') != std::string::npos) {
            // Bereich
            auto dash = arg.find('-');
            int a = std::max(1, parse_int(arg.substr(0, dash), 0));
            int b = std::max(1, parse_int(arg.substr(dash + 1), 0));
            if (a > b) std::swap(a, b);
            for (int p = a; p <= b; ++p) out.push_back(p);
        } else {
            // Komma-Liste
            std::stringstream ss(arg);
            std::string tok;
            while (std::getline(ss, tok, ',')) {
                tok = trim(tok);
                const int p = std::max(1, parse_int(tok, 0));
                if (p > 0) out.push_back(p);
            }
        }
    }

    if (out.empty() && fallback_single > 0) out.push_back(fallback_single);

    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

// ---------- main ----------

int main(int argc, char** argv) {
    // gemeinsame Optionen
    const bool serverMode   = hasFlag(argc, argv, "--server");
    const bool headless     = hasFlag(argc, argv, "--headless");
    const int  renderEvery  = std::max(1, parse_int(getOpt(argc, argv, "--render-every=", "1"), 1));
    const int  frameSkipCli = std::max(1, parse_int(getOpt(argc, argv, "--frame-skip=",  "1"), 1));
    uint64_t   seed         = parse_uint64(getOpt(argc, argv, "--seed=", "42"), 42);
    const std::string agent = getOpt(argc, argv, "--agent=", "human");
    const std::string runId = getOpt(argc, argv, "--run_id=", "cli");
    const std::string host  = getOpt(argc, argv, "--host=", "127.0.0.1");

    // Backwards-compat: alter Einzelport + neue Mehrport-Varianten
    const int         singlePort = std::max(1, parse_int(getOpt(argc, argv, "--port=", "8000"), 8000));
    const std::string portsArg   = getOpt(argc, argv, "--ports=", "");
    const int         basePort   = std::max(1, parse_int(getOpt(argc, argv, "--base-port=", "8000"), 8000));

    // Nur für lokalen Player-Modus
    const int episodes = std::max(1, parse_int(getOpt(argc, argv, "--episodes=", "1"), 1));
    const int maxSteps = std::max(0, parse_int(getOpt(argc, argv, "--max-steps=", "0"), 0));

    if (serverMode) {
        // --- Server: eine EXE, n Ports -> n Threads -> n Environments ---
        const auto ports = parse_ports(!portsArg.empty()
                                       ? portsArg
                                       : getOpt(argc, argv, "--port=", ""),
                                       singlePort, basePort);

        if (ports.size() == 1) {
            Environment env;
            env.setMode(Mode::Agent);
            env.setRenderEnabled(!headless);
            env.setRenderEvery(renderEvery);
            env.setFrameSkip(frameSkipCli);
            env.setRunInfo(agent, runId);
            env.setRenderer(nullptr);

            std::cout << "[server] starting single instance on "
                      << host << ":" << ports.front() << std::endl;
            run_server(env, host.c_str(), ports.front());
            return 0;
        }

        std::cout << "[server] starting " << ports.size() << " instances on ";
        for (size_t i = 0; i < ports.size(); ++i) std::cout << (i ? "," : "") << ports[i];
        std::cout << " (host " << host << ")" << std::endl;

        std::vector<std::unique_ptr<Environment>> envs;
        std::vector<std::thread> threads;
        envs.reserve(ports.size());
        threads.reserve(ports.size());

        for (size_t i = 0; i < ports.size(); ++i) {
            auto e = std::make_unique<Environment>();
            e->setMode(Mode::Agent);
            e->setRenderEnabled(!headless);
            e->setRenderEvery(renderEvery);
            e->setFrameSkip(frameSkipCli);

            // Run-ID pro Instanz variieren
            {
                std::ostringstream rid;
                rid << runId << "#" << i;
                e->setRunInfo(agent, rid.str());
            }
            e->setRenderer(nullptr);

            // init-capture vermeidet "assigned but never accessed"
            const int p = ports[i];
            threads.emplace_back([ptr = e.get(), host, p]() {
                run_server(*ptr, host.c_str(), p);
            });

            envs.push_back(std::move(e));
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }

        for (auto& t : threads) t.join();
        return 0;
    }

    // --- Lokaler Smoke-Test (ohne HTTP) ---
    Environment env;
    env.setMode(Mode::Player);
    env.setRenderEnabled(false);
    env.setRenderEvery(1);
    env.setFrameSkip(1);
    env.setRunInfo("human", "local");
    env.setRenderer(nullptr);

    env.reset(seed);

    for (int ep = 0; ep < episodes; ++ep) {
        int steps = 0;
        while (true) {
            auto res = env.step(1);
            ++steps;
            if (res.done) break;
            if (maxSteps > 0 && steps >= maxSteps) break;
        }
        seed += 1;
        env.reset(seed);
    }
    return 0;
}