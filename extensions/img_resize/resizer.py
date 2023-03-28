import os
from typing import List
import torch
from torchvision import transforms
from torchvision.io import read_image, write_jpeg
from PIL import Image
from tqdm.auto import tqdm


class ImageResizer:
    def __init__(self, src_dir: str, dst_dir: str, min_side: int):
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.min_side = min_side
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def resize_image(self, image_path: str, output_path: str, min_side: int) -> None:
        image = read_image(image_path)
        image = transforms.Resize(min_side)(image)
        write_jpeg(image, output_path)

"""
    把src_path的图片用Lanczos算法压缩到最小<min_size>, 保存到<dst_dir>
    <多线程版>
    :param src_dir:
    :param dst_dir:
    :param min_side: 短边分辨率
    :param img_files: src_dir里需要resize的文件名; 留空则压缩整个目录
    :return:
"""
def do_resize_images_tr(src_dir: str, dst_dir: str, min_side: int, img_files: List[str] = None) -> str:
    """

    """

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def resize_image_gpu(image_path: str, output_path: str, min_side: int) -> None:
        image = read_image(image_path).to(device)
        image = image.float()

        # Calculate new dimensions
        width, height = image.shape[-1], image.shape[-2]
        if width < height:
            new_width = min_side
            new_height = int(height * (min_side / width))
        else:
            new_height = min_side
            new_width = int(width * (min_side / height))

        # Apply Bicubic resize
        resize_transform = transforms.Resize((new_height, new_width), interpolation=Image.BICUBIC, antialias=True)
        resized_image = resize_transform(image)

        # Convert float32 back to uint8
        resized_image = torch.clamp(resized_image, 0, 255).to(torch.uint8)

        # Save the resized image
        write_jpeg(resized_image, output_path)

    if img_files is None:
        img_files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]

    for img_file in tqdm(img_files):
        input_path = os.path.join(src_dir, img_file)
        output_path = os.path.join(dst_dir, img_file)

        resize_image_gpu(input_path, output_path, min_side)

    return dst_dir

if __name__ == '__main__':
    src_dir = r"D:\Andrew\Pictures\Grabber\bench"
    dst_dir = r"D:\Andrew\Pictures\Grabber\bench.out"
    min_side = 1024
    do_resize_images_tr(src_dir, dst_dir, min_side)
