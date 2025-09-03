from two_d_env import TwoDEnvHTTP
import numpy as np

env = TwoDEnvHTTP(host="127.0.0.1", port=8000)
obs, info = env.reset(seed=42)
done = False
total = 0.0
steps = 0

while not done and steps < 200:
    a = np.random.randint(0, 5)
    obs, r, term, trunc, info = env.step(a)
    total += r
    steps += 1
    done = term or trunc

print("SMOKE OK -> steps:", steps, "return:", total)
