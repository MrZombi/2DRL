#ifndef Ghost_h
#define Ghost_h

#include "Entity.h"
#include "raylib.h"
#include <vector>
#include <random>

enum class GhostType { BLINKY, PINKY, INKY, CLYDE };

class Ghost : public Entity
{
public:
    Texture2D* normalTex;
    Texture2D* scaredTex;
    GhostType type;
    int lastDir = -1;
    int moveCounter = 0;
    int GhostDelayFrames;
    bool frightened = false;
    int frightenedTimer = 0;

    Ghost(int x, int y, Texture2D* normal, Texture2D* scared, GhostType t);

    void draw() const override;
    void move(const std::vector<std::vector<int>>& maze, const std::vector<Ghost>& ghosts, int pacmanX, int pacmanY, std::mt19937_64& rng);
    void setFrightened(int duration);
    void updateState();

};

#endif