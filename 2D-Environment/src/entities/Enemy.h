#ifndef Ghost_h
#define Ghost_h

#include "Entity.h"
#include <vector>
#include <random>

enum class GhostType { BLINKY, PINKY, INKY, CLYDE };

class Enemy : public Entity
{
public:
    GhostType type;
    int lastDir = -1;
    int moveCounter = 0;
    int GhostDelayFrames = 0;
    bool frightened = false;
    int frightenedTimer = 0;

    Enemy(int x, int y, GhostType t);

    void draw() const override;

    void move(const std::vector<std::vector<int>>& maze,
              const std::vector<Enemy>& ghosts,
              int pacmanX, int pacmanY,
              std::mt19937_64& rng);

    void setFrightened(int duration);
    void updateState();
};

#endif