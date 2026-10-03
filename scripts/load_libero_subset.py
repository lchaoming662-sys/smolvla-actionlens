from lerobot.datasets.lerobot_dataset import LeRobotDataset

EPISODES = [834, 840, 841, 845, 847, 869, 875, 878, 910, 918]
HORIZON = 50
FPS = 10
ROOT = r"E:\VLA\data\libero_actionlens"

dataset = LeRobotDataset(
    "lerobot/libero",
    root=ROOT,
    episodes=EPISODES,
    delta_timestamps={"action": [i / FPS for i in range(HORIZON)]},
)

print("root=", ROOT)
print("num_episodes=", dataset.num_episodes)
print("num_frames=", dataset.num_frames)
print("features=")
for key in dataset.features:
    print(" ", key, dataset.features[key].get("shape"), dataset.features[key].get("dtype"))

frame = dataset[0]
print("sample_keys=", sorted(frame.keys()))
print("task=", frame["task"])
print("episode_index=", frame["episode_index"].item())
print("state_shape=", tuple(frame["observation.state"].shape))
print("action_shape=", tuple(frame["action"].shape))
print("action_is_pad_shape=", tuple(frame["action_is_pad"].shape))
print("image_shape=", tuple(frame["observation.images.image"].shape))
print("image2_shape=", tuple(frame["observation.images.image2"].shape))
print("load_ok=True")
