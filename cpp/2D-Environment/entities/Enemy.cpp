#include "Enemy.h"
#include "Constants.h"
#include <random>

using Constants::Cols;
using Constants::Rows;
using Constants::CellSize;
using Constants::OffsetX;
using Constants::OffsetY;
using Constants::BlinkThreshold;
using Constants::BlinkInterval;

Enemy::Enemy(int x, int y, GhostType t)
    : Entity(x, y), type(t) {}

void Enemy::draw() const
{
    // Headless: kein Rendering
}

void Enemy::move(const std::vector<std::vector<int>>& maze,
                 const std::vector<Enemy>& /*ghosts*/,
                 int /*pacmanX*/, int /*pacmanY*/,
                 std::mt19937_64& rng)
{
    if (++moveCounter < Constants::EnemyMoveDelay) return;
    moveCounter = 0;

    static constexpr int dx[4] = {0, 1, 0, -1};
    constexpr int dy[4] = {-1, 0, 1, 0};

    const int W = static_cast<int>(maze[0].size());
    const int H = static_cast<int>(maze.size());

    auto passable = [&](int nx, int ny) {
        if (nx < 0 || nx >= W || ny < 0 || ny >= H) return false;
        int c = maze[ny][nx];
        return c != static_cast<int>(Constants::CellType::Wall);
    };

    std::vector<int> possibleDirs;
    possibleDirs.reserve(4);
    for (int dir = 0; dir < 4; ++dir) {
        // keine direkte 180°-Wende
        if ((lastDir == 0 && dir == 2) || (lastDir == 2 && dir == 0) ||
            (lastDir == 1 && dir == 3) || (lastDir == 3 && dir == 1)) {
            continue;
        }
        int nx = x + dx[dir];
        int ny = y + dy[dir];
        if (passable(nx, ny)) possibleDirs.push_back(dir);
    }

    if (possibleDirs.empty()) {
        for (int dir = 0; dir < 4; ++dir) {
            int nx = x + dx[dir], ny = y + dy[dir];
            if (passable(nx, ny)) possibleDirs.push_back(dir);
        }
        if (possibleDirs.empty()) return;
    }

    std::uniform_int_distribution<int> dist(0, static_cast<int>(possibleDirs.size()) - 1);
    int dir = possibleDirs[dist(rng)];

    x += dx[dir];
    y += dy[dir];
    lastDir = dir;
}

void Enemy::updateState()
{
    if (frightenedTimer > 0) {
        --frightenedTimer;
        frightened = true;
    } else {
        frightened = false;
    }
}

void Enemy::setFrightened(int duration)
{
    frightened      = true;
    frightenedTimer = duration;
}
