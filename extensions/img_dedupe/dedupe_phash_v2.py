import os
from typing import List
from PIL import Image
import imagehash
from tqdm import tqdm
import concurrent.futures
import sdtools.fileops as fops


class ImageDuplicateFinder:
    def __init__(self, base_dir: str, threshold: int = 5):
        self.base_dir = base_dir
        self.threshold = threshold

    def calculate_phash(self, img_path: str) -> imagehash.ImageHash:
        img = Image.open(img_path)
        # img.thumbnail((thumbnail_size, thumbnail_size), resample=Image.BICUBIC)
        return imagehash.phash(img)

    def get_artist(self, img_file):
        return img_file.split("__")[0]

    def find_duplicates(self) -> List[List[str]]:
        img_files = []
        for root, _, files in os.walk(self.base_dir):
            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
                    img_files.append(os.path.join(root, file))

        img_hashes = {}

        with concurrent.futures.ThreadPoolExecutor() as executor:
            img_hashes = dict(
                zip(
                    img_files,
                    tqdm(
                        executor.map(self.calculate_phash, img_files),
                        total=len(img_files),
                        desc="Calculating hashes",
                        unit=" images",
                    ),
                )
            )

        duplicates = []

        artist_images = {}

        for img_file in img_files:
            artist = self.get_artist(img_file)
            if artist not in artist_images:
                artist_images[artist] = []
            artist_images[artist].append(img_file)

        for artist_images_list in tqdm(
            artist_images.values(), desc="Finding duplicates", unit="pairs"
        ):
            for i, img_file1 in enumerate(artist_images_list[:-1]):
                for img_file2 in artist_images_list[i + 1 :]:
                    hash_difference = img_hashes[img_file1] - img_hashes[img_file2]
                    if hash_difference <= self.threshold:
                        # TODO: filename BUG
                        if os.path.getsize(img_file1) > os.path.getsize(img_file2):
                            duplicates.append(
                                [os.path.relpath(img_file2, self.base_dir), img_file1]
                            )
                        else:
                            duplicates.append(
                                [os.path.relpath(img_file1, self.base_dir), img_file2]
                            )

        return duplicates


def demo_run():
    image_dir = r"D:\Andrew\Pictures\Grabber\twitter_saver_2223"
    finder = ImageDuplicateFinder(image_dir, threshold=12)
    duplicates = finder.find_duplicates()
    dupes = [d[0] for d in duplicates]
    fops.move_files(image_dir, image_dir + "_duplicates", dupes, copy=False)
    print("Duplicate files:", duplicates)


if __name__ == "__main__":
    demo_run()
