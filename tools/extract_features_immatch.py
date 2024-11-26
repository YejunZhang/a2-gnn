import argparse
from argparse import Namespace
import os
import numpy as np
from tqdm import tqdm
import time
import os
import torch
from pathlib import Path
import yaml
from immatch.utils.model_helper import init_model
import cv2
import h5py

cambridge_splits = dict(
    test=["KingsCollege", "OldHospital", "ShopFacade", "StMarysChurch"],
    kings=["KingsCollege"],
    old=["OldHospital"],
    shop=["ShopFacade"],
    stmarys=["StMarysChurch"],
)


def compute_scene_im_features(detector, root_dir, dataset, split, dataset_conf):
    with open(dataset_conf, "r") as f:
        dataset_conf = Namespace(**yaml.load(f, Loader=yaml.FullLoader)[dataset])
    im_dir = Path(root_dir) / "data" / dataset_conf.data_dir
    data_processed_dir = Path(root_dir) / "data" / dataset_conf.data_processed_dir
    data_file = data_processed_dir / dataset_conf.data_file
    sids_to_load = dataset_conf.splits[split]

    # Initialize detector
    feature_cache_dir = data_processed_dir / "desc_cache" / detector.name
    feature_cache_dir.mkdir(exist_ok=True, parents=True)

    # Identify scene ids for the target split
    print(f">>>>Loading data from  {data_file} ") ## covis3-10
    data_dict = np.load(data_file, allow_pickle=True).item()

    # Load all query ids, scene ids and image data
    print(
        f"Extract features per scene, method={detector.name} cache dir={feature_cache_dir}"
    )
    for sid in tqdm(sids_to_load, total=len(sids_to_load)):
        print(sid)
        if sid not in data_dict:
            continue

        feature_path = feature_cache_dir / f"{sid}.npy"
        scene_qids = data_dict[sid]["qids"]
        scene_ims = data_dict[sid]["ims"]
        updated_count = 0
        if feature_path.exists():
            scene_features = np.load(feature_path, allow_pickle=True).item()
        else:
            scene_features = {}
        print(f"sid={sid} qids={len(scene_qids)} ims={len(scene_ims)}")

        # Extract features
        for imid, im in scene_ims.items():
            if imid not in scene_features:
                im_path = os.path.join(im_dir, im.name) # megadepth + 0000/ concrete picture.jpg
                kpts, descs = detector.load_and_extract(im_path)
                kpts = (
                    kpts.cpu().data.numpy() if isinstance(kpts, torch.Tensor) else kpts
                )# 1024,2
                descs = (
                    descs.cpu().data.numpy()
                    if isinstance(descs, torch.Tensor)
                    else descs
                ) # 1024,8
                # add color here
                colors = []
                depths = []
                original_image = cv2.imread(im_path)
                rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                
                # add depth here
                base_name = os.path.basename(im.name)
                # Split the base name and file extension
                file_name, _ = os.path.splitext(base_name) #get the img name
                
                depth_dir = '/data/zhangy36/phoenix/S6/zl548/MegaDepth_v1'
                depth_path = os.path.join(depth_dir,sid,'dense0/depths',f"{file_name}.h5")
                hdf5_file_read = h5py.File(depth_path,'r')
                gt_depth = hdf5_file_read.get('/depth')
                gt_depth = np.array(gt_depth)
                
                for kp in kpts:
                    x,y = int(kp[0]), int(kp[1])
                    r, g, b = rgb_image[y,x]
                    colors.append((r,g,b))
                    depth = gt_depth[y,x]
                    depths.append(depth)
                colors = np.array(colors)
                depths = np.array(depths)
                # import pdb; pdb.set_trace()

                hdf5_file_read.close()
                                
                scene_features[imid] = {"kpts": kpts, "descs": descs, "color": colors, "depth": depths}
                updated_count += 1
            im.kpts = scene_features[imid]["kpts"]
            im.descs = scene_features[imid]["descs"]
            im.color = scene_features[imid]["color"]
            im.depth = scene_features[imid]["depth"]

        if updated_count > 0:
            print(f"Save {updated_count} new image features. ")
            Path(feature_path).parent.mkdir(exist_ok=True, parents=True)
            np.save(feature_path, scene_features)


def extract_and_save(args):
    detector, _ = init_model(
        config=args.immatch_config, benchmark_name="default", root_dir=args.root_dir
    )
    compute_scene_im_features(
        detector, args.root_dir, args.dataset, args.split, args.dataset_config
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_config", type=str, default="configs/datasets.yml")
    parser.add_argument("--split", type=str, default="train") # this need change to generate data
    parser.add_argument("--root_dir", type=str, default=".")
    parser.add_argument("--immatch_config", type=str, default="sift") # superpoint
    parser.add_argument("--dataset", type=str, default="megadepth")
    args = parser.parse_args()
    print(args)
    extract_and_save(args)
