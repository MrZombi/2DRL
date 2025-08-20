#include "Pacman.h"
#include "raylib.h"
#include "Constants.h"
#include "Action.h"

using Constants::CellSize;
using Constants::OffsetX;
using Constants::OffsetY;
using Constants::Cols;
using Constants::Rows;

constexpr Color pacmanYellow   = YELLOW;

Pacman::Pacman(int x, int y) : Entity(x, y){}

void Pacman::updateAnimation()
{
    if (animationTimer > 0)
    {
        float progress = static_cast<float>(animationTimer) / animationDuration;
        if (mouthOpening)
        {
            mouthOpenAngle = 60.0f * progress;
        }
        else
        {
            mouthOpenAngle = 60.0f * (1.0f - progress);
        }

        animationTimer--;
        if (animationTimer == animationDuration / 2)
        {
            mouthOpening = !mouthOpening;
        }
    }
    else
    {
        mouthOpenAngle = 60.0f;
    }
}

void Pacman::onCoinCollected()
{
    animationTimer = animationDuration;
    mouthOpening = true;
}

void Pacman::draw() const
{
    int cx = OffsetX + x * CellSize + CellSize / 2;
    int cy = OffsetY + y * CellSize + CellSize / 2;
    float angle = 0.0f;
    if (dirX == 1 && dirY == 0) angle = 0.0f;
    else if (dirX == -1 && dirY == 0) angle = 180.0f;
    else if (dirX == 0 && dirY == -1) angle = 270.0f;
    else if (dirX == 0 && dirY == 1) angle = 90.0f;
    float startAngle = angle + mouthOpenAngle / 2;
    float endAngle = angle - mouthOpenAngle / 2 + 360.0f;
    DrawCircleSector(
        { static_cast<float>(cx),
        static_cast<float>(cy) },
        static_cast<float>(CellSize) * 0.5f - 2.0f,
        startAngle,endAngle,32,pacmanYellow
    );

}

void Pacman::handleInput(const std::vector<std::vector<int>>& maze)
{
    int nx = 0, ny = 0;
    bool pressed = false;

    if (IsKeyPressed(KEY_UP) || IsKeyPressed(KEY_W))    { nx = x; ny = y - 1; dirX = 0; dirY = -1; pressed = true; }
    else if (IsKeyPressed(KEY_DOWN) || IsKeyPressed(KEY_S)) { nx = x; ny = y + 1; dirX = 0; dirY = 1; pressed = true; }
    else if (IsKeyPressed(KEY_LEFT) || IsKeyPressed(KEY_A)) { nx = x - 1; ny = y; dirX = -1; dirY = 0; pressed = true; }
    else if (IsKeyPressed(KEY_RIGHT) || IsKeyPressed(KEY_D)) { nx = x + 1; ny = y; dirX = 1; dirY = 0; pressed = true; }

    if (pressed)
    {
        if (nx < 0) nx = Cols - 1;
        if (nx >= Cols) nx = 0;
        if (ny >= 0 && ny < Rows && maze[ny][nx] != 1 && maze[ny][nx] != 3)
        {
            x = nx; y = ny;
            moveCounter = Constants::PacmanMoveDelay;
            keyHeld = true;
            return;
        }
    }
}

void Pacman::applyAction(Action a, const std::vector<std::vector<int>>& maze)
{
    if (moveCounter > 0) { --moveCounter; return; }

    int nx = x; int ny = y;
    int dx = 0, dy = 0;
    switch (a)
        {
            case Action::Up:    dx =  0; dy = -1; break;
                case Action::Right: dx =  1; dy =  0; break;
                case Action::Down:  dx =  0; dy =  1; break;
                case Action::Left:  dx = -1; dy =  0; break;
                case Action::Stay:  dx =  0; dy =  0; break;
        }
    dirX = dx; dirY = dy;
    nx = x + dx; ny = y + dy;

    if (nx < 0) nx = Cols - 1;
    if (nx >= Cols) nx = 0;

    if (ny >= 0 && ny < Rows)
        {
            int cell = maze[ny][nx];
            if (cell != static_cast<int>(Constants::CellType::Wall) &&
                cell != static_cast<int>(Constants::CellType::Gate)) {
                    x = nx; y = ny;
                    moveCounter = Constants::PacmanMoveDelay;
                }
        }
}