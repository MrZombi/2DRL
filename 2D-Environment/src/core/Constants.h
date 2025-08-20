#pragma once

namespace Constants
{
    constexpr int Cols          = 23;
    constexpr int Rows          = 23;
    constexpr int CellSize      = 45;
    constexpr int OffsetX       = 28;
    constexpr int OffsetY       = 28;

    constexpr int PowerDuration    = 540;
    constexpr int PowerRespawnTime = 1080;

    constexpr int FruitRespawnTime  = 1080;

    constexpr int PickupActiveTime  = 200;

    constexpr int BlinkThreshold    = 180;
    constexpr int BlinkInterval     = 15;

    enum class CellType : int
    {
        Empty = 0,
        Wall  = 1,
        Coin  = 2,
        Gate = 3
    };

    constexpr int GameFPS = 60;
    constexpr int GhostMoveDelay = 10;
    constexpr int PacmanMoveDelay = 8;
}