# description: create down-sampled image from existing Your Name frame data


import argparse
import os
from PIL import Image

from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Down-sample Your Name frame.")

    parser.add_argument("--src", type=int, help="resolution of source video (1080/480)")
    parser.add_argument("--dst", type=int, help="resolution of target video, should be dividable")
    parser.add_argument("--start", type=int, default=1, help="start index of frame")
    parser.add_argument("--end", type=int, default=159661, help="end index of frame")
    args = parser.parse_args()

    assert (args.src % args.dst == 0)
    assert (args.dst % 2 == 0)
    assert (1 <= args.start <= args.end <= 159661)

    src_image_dir = os.path.join("/data", f"{args.src}")
    dst_image_dir = os.path.join("/data", f"{args.dst}")

    assert (os.path.exists(src_image_dir))

    if not os.path.exists(dst_image_dir):
        os.makedirs(dst_image_dir)

    images = sorted(os.listdir(src_image_dir))[args.start-1:args.end]
    # images = sorted(os.listdir(src_image_dir))

    dst_height = args.dst
    dst_width = args.dst // 2 * 3

    pbar = tqdm(images)
    pbar.set_description(f"from {args.start} to {args.end}")
    for image in pbar:
        src_image_path = os.path.join(src_image_dir, image)
        src_image = Image.open(src_image_path).convert('RGB')
        dst_image = src_image.resize((dst_width, dst_height))
        dst_image_path = os.path.join(dst_image_dir, image)
        dst_image.save(dst_image_path)
