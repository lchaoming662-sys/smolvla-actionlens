from lerobot.datasets.lerobot_dataset import LeRobotDataset

ALL_EPISODES = [834, 840, 841, 845, 847, 869, 875, 878, 910, 918,
                920, 929, 944, 948, 949, 953, 966, 972, 975, 977]
TRAIN_EPISODES = [834, 840, 841, 845, 847, 869, 875, 878, 966, 972, 975, 977]
TEST_EPISODES = [910, 918, 920, 929, 944, 948, 949, 953]
ROOT = r"E:\VLA\data\libero_actionlens_20"

dataset = LeRobotDataset(
    "lerobot/libero",
    root=ROOT,
    episodes=ALL_EPISODES,
    delta_timestamps={"action": [i / 10 for i in range(50)]},
)
print("root=", ROOT)
print("all_episodes=", ALL_EPISODES)
print("train_episodes=", TRAIN_EPISODES)
print("test_episodes=", TEST_EPISODES)
print("num_episodes=", dataset.num_episodes)
print("num_frames=", dataset.num_frames)
frame = dataset[0]
print("task=", frame["task"])
print("action_shape=", tuple(frame["action"].shape))
print("action_is_pad_shape=", tuple(frame["action_is_pad"].shape))
print("image_shape=", tuple(frame["observation.images.image"].shape))
print("image2_shape=", tuple(frame["observation.images.image2"].shape))
print("dataset20_ok=True")
