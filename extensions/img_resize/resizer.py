import os
from typing import List
import torch
from torchvision import transforms
from torchvision.io import read_image, write_jpeg
from PIL import Image
from tqdm.auto import tqdm


"""
===
STATE: NOT FASTER THAN CPU, use gradio version instead
===
"""
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



import os
from typing import List
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, write_jpeg
from torchvision.transforms import Resize
from tqdm import tqdm

class ImageDataset(Dataset):
    def __init__(self, src_dir: str, img_files: List[str]):
        self.src_dir = src_dir
        self.img_files = img_files

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.src_dir, self.img_files[idx])
        image = read_image(img_path)
        return image, self.img_files[idx]

class GPUImageResizer:
    def __init__(self, in_dir: str, out_dir: str, min_side: int):
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.min_side = min_side

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

    def resize_images(self):
        img_files = [f for f in os.listdir(self.in_dir) if os.path.isfile(os.path.join(self.in_dir, f))]
        dataset = ImageDataset(self.in_dir, img_files)
        batch_size = 8
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        for batch in tqdm(dataloader, total=len(dataloader)):
            images, image_filenames = batch
            images = images.to(device)

            resized_images = self.batch_resize(images)

            for i in range(images.size(0)):
                output_path = os.path.join(self.out_dir, image_filenames[i])
                write_jpeg(resized_images[i].cpu(), output_path)

    def batch_resize(self, images: torch.Tensor) -> torch.Tensor:
        images = images.float()

        # Calculate new dimensions
        widths, heights = images.shape[-1], images.shape[-2]
        ratios = self.min_side / torch.minimum(widths, heights)
        new_widths = (widths * ratios).int()
        new_heights = (heights * ratios).int()

        # Apply Bicubic resize
        resize_transform = Resize((new_heights, new_widths), interpolation=3)
        resized_images = resize_transform(images)

        # Convert float32 back to uint8
        resized_images = torch.clamp(resized_images, 0, 255).to(torch.uint8)

        return resized_images



def try_gpu():
    in_dir = r"D:\Andrew\Pictures\Grabber\bench"
    out_dir = r"D:\Andrew\Pictures\Grabber\bench.out"
    min_side = 256
    resizer = GPUImageResizer(in_dir, out_dir, min_side)
    resizer.resize_images()


def try_cpu():
    src_dir = r"D:\Andrew\Pictures\Grabber\bench"
    dst_dir = r"D:\Andrew\Pictures\Grabber\bench.out"
    min_side = 1024
    do_resize_images_tr(src_dir, dst_dir, min_side)

if __name__ == '__main__':
    try_cpu()
    # try_cpu()

