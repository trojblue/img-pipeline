import os
from typing import List
from PIL import Image
import imagehash
from collections import defaultdict
import mimetypes
from tqdm import tqdm
import concurrent.futures

class DuplicateImageRemover:
    def __init__(self, image_dir: str, threshold: int = 5):
        self.image_dir = image_dir
        self.threshold = threshold


    def find_duplicates(self) -> List[str]:
        img_files = [f for f in os.listdir(self.image_dir) if os.path.isfile(os.path.join(self.image_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        img_hashes = {}
        duplicates = []

        for img_file in tqdm(img_files, desc="Calculating hashes", unit=" images"):
            img_path = os.path.join(self.image_dir, img_file)
            img_hash = self.calculate_phash(img_path)
            img_hashes[img_file] = img_hash

        for i, img_file1 in tqdm(enumerate(img_files[:-1]), desc="Comparing hashes", unit=" pairs"):
            for img_file2 in img_files[i + 1:]:
                hash_difference = img_hashes[img_file1] - img_hashes[img_file2]
                if hash_difference <= self.threshold:
                    duplicates.append(img_file2)

        return duplicates

    def find_duplicates2(self) -> List[str]:
        image_files = [f for f in os.listdir(self.image_dir) if self.is_image(os.path.join(self.image_dir, f))]
        hash_dict = defaultdict(list)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            img_hashes = list(tqdm(executor.map(self.calculate_phash, [os.path.join(self.image_dir, img_file) for img_file in image_files]), total=len(image_files), desc="Calculating image hashes"))

        for img_file, img_hash in zip(image_files, img_hashes):
            hash_dict[str(img_hash)].append(img_file)

        duplicates = self.compare_hashes(hash_dict)

        return duplicates

    def is_image(self, file_path: str) -> bool:
        mimetype, _ = mimetypes.guess_type(file_path)
        return mimetype and mimetype.startswith("image")

    def calculate_phash(self, img_path: str) -> imagehash.ImageHash:
        img = Image.open(img_path)
        img_hash = imagehash.phash(img)
        return img_hash

    def compare_hashes(self, hash_dict) -> List[str]:
        duplicates = []

        for img_hash, img_list in hash_dict.items():
            if len(img_list) > 1:
                ref_hash = imagehash.hex_to_hash(img_hash)

                for img_file in img_list[1:]:
                    current_hash = self.calculate_phash(os.path.join(self.image_dir, img_file))
                    if ref_hash - current_hash <= self.threshold:
                        duplicates.append(img_file)

        return duplicates

if __name__ == "__main__":
    image_dir = r"D:\Andrew\Pictures\Grabber\twitter_saver_2023"
    threshold = 500

    remover = DuplicateImageRemover(image_dir, threshold)
    duplicates = remover.find_duplicates()

    print("Duplicate image files:", duplicates)
