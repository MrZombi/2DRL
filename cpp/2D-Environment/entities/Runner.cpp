#include "Runner.h"
#include "Constants.h"
#include "Action.h"

using Constants::CellSize;
using Constants::OffsetX;
using Constants::OffsetY;
using Constants::Cols;
using Constants::Rows;

Runner::Runner(int x, int y) : Entity(x, y) {}

void Runner::updateAnimation()
{
    if (animationTimer > 0)
        {
            float progress = static_cast<float>(animationTimer) / animationDuration;
            mouthOpenAngle = mouthOpening ? (60.0f * progress) : (60.0f * (1.0f - progress));
            animationTimer--;
            if (animationTimer == animationDuration / 2) mouthOpening = !mouthOpening;
        }
    else
        {
            mouthOpenAngle = 60.0f;
        }
}

void Runner::onCoinCollected()
{
    animationTimer = animationDuration;
    mouthOpening = true;
}

void Runner::draw() const
{
    // Headless: kein Rendering
}

void Runner::handleInput(const std::vector<std::vector<int>>& /*maze*/)
{
    // Headless: keine Tastatur; Eingaben kommen über Environment::setPendingAction()
}

void Runner::applyAction(Action a, const std::vector<std::vector<int>>& maze)
{
    if (moveCounter > 0) { --moveCounter; return; }

    int nx; int ny;
    int dx = 0, dy = 0;
    switch (a) {
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

    if (ny >= 0 && ny < Rows) {
            int cell = maze[ny][nx];
            if (cell != static_cast<int>(Constants::CellType::Wall) &&
                cell != static_cast<int>(Constants::CellType::Gate)) {
                    x = nx; y = ny;
                    moveCounter = Constants::RunnerMoveDelay;
                }
    }
}