#ifndef Entity_h
#define Entity_h

class Entity
{
public:
    int x, y;
    Entity(int x, int y);

    virtual void draw() const = 0;

    virtual ~Entity() = default;
};

inline bool checkCollision(const Entity& a, const Entity& b)
{
    return a.x == b.x && a.y == b.y;
}

#endif