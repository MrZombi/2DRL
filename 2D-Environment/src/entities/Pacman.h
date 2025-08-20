#ifndef Pacman_h
#define Pacman_h

#include "Entity.h"
#include <vector>
#include "Action.h"

class Pacman : public Entity
{
public:
    int score = 0;
    int dirX = 1, dirY = 0;
    int moveCounter = 0;
    int PacmanMoveDelay;
    bool keyHeld = false;
    float mouthOpenAngle = 60.0f;
    bool mouthOpening = false;
    int animationTimer = 0;
    static constexpr int animationDuration = 10;

    //Konstruktor
    Pacman(int x, int y);

    void onCoinCollected();
    void updateAnimation();
    void draw() const override;
    void handleInput(const std::vector<std::vector<int>>& maze);
    void applyAction(Action a, const std::vector<std::vector<int>>& maze);
};

#endif