#include "Renderer.h"
#include "raylib.h"
#include "Constants.h"

Renderer::~Renderer() {
    if (initialized_) {
        unloadSprites_();
        CloseWindow();
    }
}

void Renderer::initIfNeeded() {
    if (initialized_) return;
    constexpr int width  = Constants::OffsetX * 2 + Constants::Cols * Constants::CellSize;
    constexpr int height = Constants::OffsetY * 2 + Constants::Rows * Constants::CellSize;
    InitWindow(width, height, "Pacman RL");
    SetTargetFPS(60);
    initialized_ = true;

    loadSprites_();
}

void Renderer::shutdown() {
    if (initialized_) {
        unloadSprites_();
        CloseWindow();
        initialized_ = false;
    }
}

void Renderer::loadSprites_() {
    if (spritesLoaded_) return;
    // Pfade wie in deinem alten GameView
    texGhostRed_    = LoadTexture("resources/ghost_red.png");
    texGhostPink_   = LoadTexture("resources/ghost_pink.png");
    texGhostCyan_   = LoadTexture("resources/ghost_cyan.png");
    texGhostOrange_ = LoadTexture("resources/ghost_orange.png");
    texGhostScared_ = LoadTexture("resources/ghost_scared.png");
    texCherry_      = LoadTexture("resources/cherry.png");
    spritesLoaded_  = true;
}

void Renderer::unloadSprites_() {
    if (!spritesLoaded_) return;
    if (texGhostRed_.id)    UnloadTexture(texGhostRed_);
    if (texGhostPink_.id)   UnloadTexture(texGhostPink_);
    if (texGhostCyan_.id)   UnloadTexture(texGhostCyan_);
    if (texGhostOrange_.id) UnloadTexture(texGhostOrange_);
    if (texGhostScared_.id) UnloadTexture(texGhostScared_);
    if (texCherry_.id)      UnloadTexture(texCherry_);
    spritesLoaded_ = false;
}

void Renderer::draw(const RenderState& s) {
    if (!s.maze || !s.pacman || !s.ghosts) return;
    initIfNeeded();

    BeginDrawing();
    ClearBackground(BLACK);

    for (int y = 0; y < static_cast<int>(s.maze->size()); ++y) {
        for (int x = 0; x < static_cast<int>((*s.maze)[y].size()); ++x) {
            int cell = (*s.maze)[y][x];
            const int px = Constants::OffsetX + x * Constants::CellSize;
            const int py = Constants::OffsetY + y * Constants::CellSize;

            if (cell == static_cast<int>(Constants::CellType::Wall)) {
                DrawRectangle(px, py, Constants::CellSize, Constants::CellSize, DARKBLUE);
                DrawRectangleLines(px, py, Constants::CellSize, Constants::CellSize, BLACK);
            } else if (cell == static_cast<int>(Constants::CellType::Coin)) {
                DrawCircle(px + Constants::CellSize/2,
                           py + Constants::CellSize/2,
                           Constants::CellSize * 0.15f, YELLOW);
            } else if (cell == static_cast<int>(Constants::CellType::Gate)) {
                DrawRectangle(px, py, Constants::CellSize, 4, GRAY);
            }
        }
    }

    // Pickups
    if (s.pillActive) {
        const int px = Constants::OffsetX + s.pillX * Constants::CellSize + Constants::CellSize/2;
        const int py = Constants::OffsetY + s.pillY * Constants::CellSize + Constants::CellSize/2;
        DrawCircle(px, py, Constants::CellSize * 0.25f, ORANGE);
    }
    if (s.fruitActive && texCherry_.id) {
        const float fx = Constants::OffsetX + s.fruitX * Constants::CellSize + Constants::CellSize * 0.5f;
        const float fy = Constants::OffsetY + s.fruitY * Constants::CellSize + Constants::CellSize * 0.5f;
        DrawTexturePro(
            texCherry_,
            Rectangle{0.0f, 0.0f, static_cast<float>(texCherry_.width), static_cast<float>(texCherry_.height)},
            Rectangle{fx - Constants::CellSize * 0.5f, fy - Constants::CellSize * 0.5f,
                      static_cast<float>(Constants::CellSize), static_cast<float>(Constants::CellSize)},
            Vector2{0.0f, 0.0f},
            0.0f,
            WHITE
        );
    }

    {
        const int cx = Constants::OffsetX + s.pacman->x * Constants::CellSize + Constants::CellSize/2;
        const int cy = Constants::OffsetY + s.pacman->y * Constants::CellSize + Constants::CellSize/2;
        DrawCircle(cx, cy, Constants::CellSize * 0.45f, GOLD);
    }

    for (size_t i = 0; i < s.ghosts->size(); ++i) {
        const auto& g = (*s.ghosts)[i];
        const float gx = Constants::OffsetX + g.x * Constants::CellSize + Constants::CellSize * 0.5f;
        const float gy = Constants::OffsetY + g.y * Constants::CellSize + Constants::CellSize * 0.5f;

        const Texture2D* tex = nullptr;
        if (g.frightened && texGhostScared_.id)
            {
            tex = &texGhostScared_;
            }
        else
            {
            const size_t idx = i % 4;
            if      (idx == 0 && texGhostRed_.id)    tex = &texGhostRed_;
            else if (idx == 1 && texGhostPink_.id)   tex = &texGhostPink_;
            else if (idx == 2 && texGhostCyan_.id)   tex = &texGhostCyan_;
            else if (idx == 3 && texGhostOrange_.id) tex = &texGhostOrange_;
            }

        if (tex) {
            DrawTexturePro(
                *tex,
                Rectangle{0.0f, 0.0f, static_cast<float>(tex->width), static_cast<float>(tex->height)},
                Rectangle{gx - Constants::CellSize * 0.5f, gy - Constants::CellSize * 0.5f,
                          static_cast<float>(Constants::CellSize), static_cast<float>(Constants::CellSize)},
                Vector2{0.0f, 0.0f},
                0.0f,
                WHITE
            );
        } else {
            DrawCircle(static_cast<int>(gx), static_cast<int>(gy), Constants::CellSize * 0.4f, g.frightened ? SKYBLUE : RED);
        }
    }

    DrawText(TextFormat("Score: %d", s.score), 12, 8, 18, RAYWHITE);
    if (s.powerTimer > 0) DrawText("POWER!", 200, 8, 18, LIME);
    if (s.gameOver)       DrawText("GAME OVER", 320, 8, 18, RED);

    EndDrawing();
}
