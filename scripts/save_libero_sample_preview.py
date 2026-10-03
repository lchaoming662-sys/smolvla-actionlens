from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

EPISODES = [834, 840, 841, 845, 847, 869, 875, 878, 910, 918]
ROOT = r"E:\VLA\data\libero_actionlens"
ds = LeRobotDataset("lerobot/libero", root=ROOT, episodes=EPISODES)
frame = ds[0]

def to_pil(x):
    arr = x.detach().cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

img1 = to_pil(frame["observation.images.image"])
img2 = to_pil(frame["observation.images.image2"])
canvas = Image.new("RGB", (512, 300), "white")
canvas.paste(img1.resize((256, 256)), (0, 30))
canvas.paste(img2.resize((256, 256)), (256, 30))
draw = ImageDraw.Draw(canvas)
draw.text((8, 8), "LIBERO sample 0 | top + wrist cameras", fill="black")
out = Path(r"E:\VLA\outputs\libero_subset_sample.png")
out.parent.mkdir(parents=True, exist_ok=True)
canvas.save(out)
print("saved=", out)
print("image_size=", canvas.size)
