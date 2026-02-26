# from openpi.training.rlds_dataloader_final.rlds_dataset import RLDSDataset
# from torch.utils.data import DataLoader
# dataset = RLDSDataset(
#         data_root_dir = "/home/yajatyadav/tensorflow_datasets/",
#         data_mix = "franka",
#         shuffle_buffer_size=100_000,
#     )

# dataloader = DataLoader(
#         dataset,
#         batch_size=32,
#         sampler=None,
#         collate_fn=None,
#         num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
#     )

# import pdb; pdb.set_trace()

from openpi.training.rlds_dataloader.dataloader import create_data_loader, create_dataset, transform_dataset
from openpi.training.config import get_config
import openpi.training.sharding as sharding
import jax
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
from openpi.models import model as _model
import typing
import torch
import random
import os
import matplotlib.pyplot as plt

def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *x: np.stack(np.asarray(x), axis=0), *items)


def save_debug_image(image, idx: int, dir_name: str = "rlds_dataset_images"):
    """Helper function to save images for debugging.
    
    Args:
        image: Image array to save
        idx: Index for filename
        dir_name: Directory to save images to
    """
    ## reshape image (C, H, W) -> (H, W, C)
    # image = image.detach().cpu().permute(1, 2, 0).numpy()
    # image = np.transpose(image, (1, 2, 0))
    # image = image.astype(np.uint8)

    os.makedirs(dir_name, exist_ok=True)
    random_suffix = random.randint(0, 1000000)  # Generate a random 6-digit number
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(os.path.join(dir_name, f"image_{idx}_{random_suffix}.png"))
    plt.close()



config = get_config("full_FT_whiteboard_RLDS")
mesh = sharding.make_mesh(config.fsdp_devices)
data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
data_config = config.data.create(config.assets_dirs, config.model)




dataset = create_dataset(config, config.model)
# dataset = transform_dataset(dataset, data_config)
 
for sample in dataset:
    print(sample)
    # import pdb; pdb.set_trace()
#     save_debug_image(sample["observation"]["image_primary"].squeeze(0), 0)
#     save_debug_image(sample["observation"]["image_wrist"].squeeze(0), 1)

# for batch in dataloader:
#     import pdb; pdb.set_trace()
#     obs, action = _model.Observation.from_dict(batch), batch["actions"]

# dataloader = create_data_loader(config, sharding=data_sharding, num_workers=config.num_workers)
# print
# for batch in dataloader:
    # import pdb; pdb.set_trace()



# dataloader = create_data_loader(
#         config,
#         sharding=data_sharding,
#         num_workers=0,
#         shuffle=True,
#     )
# i = 0
# approx_num_batches = 27044326 // config.batch_size
# visualize_every = 1000
# for batch in tqdm(dataloader):
#     i += 1
#     if i % approx_num_batches == 0:
#         print("using the dataloader, finished one epoch!!")
#     if i % visualize_every == 0:
#         print("TODO: some visualization logic to sanity check the model inputs")
#         # plt.save()
