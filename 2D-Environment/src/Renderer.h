#pragma once
#include <vector>
#include "raylib.h"
#include "Pacman.h"
#include "Ghost.h"

struct RenderState {
    const std::vector<std::vector<int>>* maze = nullptr;
    const Pacman* pacman = nullptr;
    const std::vector<Ghost>* ghosts = nullptr;
    int  powerTimer = 0;
    bool gameOver = false;
    int  score = 0;

    bool pillActive  = false; int pillX  = -1, pillY  = -1;
    bool fruitActive = false; int fruitX = -1, fruitY = -1;
};

class Renderer {
public:
    Renderer() = default;
    ~Renderer();

    void initIfNeeded();
    void shutdown();

    void draw(const RenderState& s);

private:
    bool initialized_ = false;

    // --- Sprites ---
    bool spritesLoaded_ = false;
    Texture2D texGhostRed_{};
    Texture2D texGhostPink_{};
    Texture2D texGhostCyan_{};
    Texture2D texGhostOrange_{};
    Texture2D texGhostScared_{};
    Texture2D texCherry_{};

    void loadSprites_();
    void unloadSprites_();
};