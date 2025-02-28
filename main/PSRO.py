from classEnv import GameEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from graphClass import Graph
import gymnasium

def main(graph):
    # 1) Create the environment (with mode='human' if you want to see the GUI)
    env = GameEnv(graph, mode=None)
    
    # 2) Wrap the environment with ActionMasker so invalid actions are masked
    env = ActionMasker(env, mask_fn)

    # 3) Create and train the model
    model = MaskablePPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)

    # 4) Save the model
    model.save("Mask_10k")

    # 5) Load the model (you can load into the same env or a new one)
    loaded_model = MaskablePPO.load("Mask_10k", env=env)

    # 6) Manually run a few episodes to see how the model plays
    num_episodes = 3
    env.render()
    for episode in range(num_episodes):
        print(f"--- Episode {episode+1} ---")
        obs, info = env.reset()
        done = False
        while not done:
            # We still want to mask invalid actions before predicting
            action_masks = env.action_masks()
            action, _states = loaded_model.predict(obs, deterministic=True, action_masks=action_masks)
            obs, reward, done, truncated, info = env.step(action)
            # Render the environment (if you want to see the GUI updates)
            if done or truncated:
                print(f"Episode {episode+1} finished with reward: {reward}")
                break

    # 7) Close the environment (closes any GUI windows)
    env.close()

def show(graph):
    # 1) Create the environment (with mode='human' if you want to see the GUI)
    env = GameEnv(graph, mode=None)
    
    # 2) Wrap the environment with ActionMasker so invalid actions are masked
    env = ActionMasker(env, mask_fn)

    # 5) Load the model (you can load into the same env or a new one)
    loaded_model = MaskablePPO.load("Mask_10k", env=env)

    # 6) Manually run a few episodes to see how the model plays
    num_episodes = 3
    env.render()
    for episode in range(num_episodes):
        print(f"--- Episode {episode+1} ---")
        obs, info = env.reset()
        done = False
        while not done:
            # We still want to mask invalid actions before predicting
            action_masks = env.action_masks()
            action, _states = loaded_model.predict(obs, deterministic=True, action_masks=action_masks)
            obs, reward, done, truncated, info = env.step(action)
            # Render the environment (if you want to see the GUI updates)
            if done or truncated:
                print(f"Episode {episode+1} finished with reward: {reward}")
                break

    # 7) Close the environment (closes any GUI windows)
    env.close()

def mask_fn(env: gymnasium.Env):
    return env.action_mask()

if __name__ == "__main__":
    file = '/Users/davidklasa/Documents/KTH/KTH Kandidatexamensjobb/Kod/graph1.csv'
    graph = Graph(file)
    # to train and show how plays
    #main(graph)
    # to show how plays with previously trained model
    show(graph)