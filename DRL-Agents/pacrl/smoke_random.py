import time
from pacrl.env import Arcade2DEnv

def main():
    env = Arcade2DEnv(server_url="http://127.0.0.1:8000",
                      seed=123, render=False, frame_skip=1, render_every=1,
                      agent_name="SmokeTest", run_id=int(time.time()))
    obs, _ = env.reset()
    print("obs shape:", obs.shape)

    total = 0.0
    for t in range(200):
        a = env.action_space.sample()
        obs, r, term, trunc, _ = env.step(a)
        total += r
        if term or trunc:
            print(f"episode done @ {t}, return={total:.2f}")
            obs, _ = env.reset(); total = 0.0
    print("done.")

if __name__ == "__main__":
    main()