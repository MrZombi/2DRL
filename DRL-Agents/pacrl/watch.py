# watch.py
import os, argparse, time
from stable_baselines3 import DQN, PPO, A2C
from pacrl.env import Arcade2DEnv

ALGOS = {"dqn": DQN, "ppo": PPO, "a2c": A2C}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=list(ALGOS.keys()), required=True)
    ap.add_argument("--model", required=True, help="path to .zip saved model")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--render-every", type=int, default=1)
    ap.add_argument("--frame-skip", type=int, default=1)
    args = ap.parse_args()

    EnvCls = Arcade2DEnv
    # Headless=False, damit C++-Fenster rendert
    env = EnvCls(server_url=os.environ.get("ARCADE2D_SERVER", "http://127.0.0.1:8000"),
                 seed=123, render=True, frame_skip=args.frame_skip,
                 render_every=args.render_every, agent_name="Playback", run_id=int(time.time()))

    ModelCls = ALGOS[args.algo]
    model = ModelCls.load(args.model)  # Laden ohne VecEnv (wir nutzen .predict direkt)

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        ret = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            if "r_extrinsic" in info and "r_intrinsic" in info:
                pass
            ret += reward
            done = terminated or truncated
        print(f"Episode {ep+1}/{args.episodes} return={ret:.2f}")

if __name__ == "__main__":
    main()