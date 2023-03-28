import os
from typing import List
from PIL import Image
import imagehash
from tqdm import tqdm
import concurrent.futures
import sdtools.fileops as fops


class ImageDuplicateFinder:
    def __init__(self, image_dir: str, threshold: int = 5):
        self.image_dir = image_dir
        self.threshold = threshold

    def calculate_phash(self, img_path: str) -> imagehash.ImageHash:
        img = Image.open(img_path)
        return imagehash.phash(img)

    def compare_hashes(self, img_hashes, img_file1, img_file2):
        hash_difference = img_hashes[img_file1] - img_hashes[img_file2]
        if hash_difference <= self.threshold:
            return img_file2
        else:
            return None

    def get_artist(self, img_file):
        return img_file.split("__")[0]

    def find_duplicates(self) -> List[List[str]]:
        img_files = [f for f in os.listdir(self.image_dir) if
                     os.path.isfile(os.path.join(self.image_dir, f)) and f.lower().endswith(
                         ('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        img_hashes = {}

        with concurrent.futures.ThreadPoolExecutor() as executor:
            img_paths = [os.path.join(self.image_dir, img_file) for img_file in img_files]
            img_hashes = dict(zip(img_files, tqdm(executor.map(self.calculate_phash, img_paths), total=len(img_paths),
                                                  desc="Calculating hashes", unit=" images")))

        duplicates = []

        for i, img_file1 in tqdm(enumerate(img_files[:-1]), desc="Comparing hashes", unit=" pairs"):
            for img_file2 in img_files[i + 1:]:
                artist = self.get_artist(img_file1)
                artist2 = self.get_artist(img_file2)
                if artist == artist2:
                    hash_difference = img_hashes[img_file1] - img_hashes[img_file2]
                    if hash_difference <= self.threshold:
                        duplicates.append([img_file2, img_file1])

        return duplicates


def demo_run():
    image_dir =  r"D:\Andrew\Pictures\Grabber\twitter_saver_2023"
    finder = ImageDuplicateFinder(image_dir, threshold=12)
    duplicates = finder.find_duplicates()
    dupes = [d[0] for d in duplicates]
    fops.move_files(image_dir, image_dir+"_duplicates", dupes, copy=True)
    print("Duplicate files:", duplicates)

if __name__ == "__main__":
    demo_run()
