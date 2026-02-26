from openpi.training.data_loader import create_data_loader, create_dataset, transform_dataset
from openpi.training.config import get_config
import openpi.training.sharding as sharding
import jax
from tqdm import tqdm
import numpy as np
import os
import random
import matplotlib.pyplot as plt

def save_debug_image(image, idx: int, dir_name: str = "lerobot_dataset_images"):
    """Helper function to save images for debugging.
    
    Args:
        image: Image array to save
        idx: Index for filename
        dir_name: Directory to save images to
    """
    ## reshape image (C, H, W) -> (H, W, C)
    image = image.detach().cpu().permute(1, 2, 0).numpy()
    # image = image.astype(np.uint8)

    os.makedirs(dir_name, exist_ok=True)
    random_suffix = random.randint(0, 1000000)  # Generate a random 6-digit number
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(os.path.join(dir_name, f"image_{idx}_{random_suffix}.png"))
    plt.close()



config = get_config("full_FT_whiteboard")
mesh = sharding.make_mesh(config.fsdp_devices)
data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
data_config = config.data.create(config.assets_dirs, config.model)
dataset = create_dataset(data_config, config.model)
# dataset = transform_dataset(dataset, data_config)
# for sample in dataset:
    # import pdb; pdb.set_trace()
    # save_debug_image(sample["exterior_image_1"], 0)
    # save_debug_image(sample["wrist_image"], 1)
    # break


dataloader = create_data_loader(
        config,
        sharding=data_sharding,
        num_workers=config.num_workers,
        shuffle=True,
    )
for batch in dataloader:
    # import pdb; pdb.set_trace()
    break

# base_torch_dataloader = dataloader._data_loader._data_loader
# dataset = base_torch_dataloader.dataset
# batch_size = base_torch_dataloader.batch_size
# # for batch in tqdm(base_torch_dataloader, total=len(dataset) // batch_size):
#     # import pdb; pdb.set_trace()
#     # break

# for batch in dataloader:
#     x = batch
#     import pdb; pdb.set_trace()
#     break