import os
import csv
from tqdm import tqdm
from PIL import Image
import imagehash
import numpy as np
from scipy.spatial.distance import hamming

class ImageDuplicateRemover:
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def _load_images(self):
        """Load all images from the folder."""
        self.image_paths = []
        self.images = []
        for file_name in os.listdir(self.folder_path):
            if file_name.endswith('.jpg') or file_name.endswith('.png'):
                file_path = os.path.join(self.folder_path, file_name)
                self.image_paths.append(file_path)
                image = Image.open(file_path)
                self.images.append(image)

    def _calculate_hashes(self):
        """Calculate pHash values for all images."""
        self.hashes = []
        for image in self.images:
            hash_value = imagehash.phash(image, hash_size=16)
            hash_str = str(hash_value)
            self.hashes.append(hash_str)

        with open('hashes.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows([hash_str] for hash_str in self.hashes)

    def _find_duplicates(self):
        """Find and remove duplicate images."""
        self.duplicates = set()
        self.hashes_binary = np.array([int(str(hash), 16) for hash in self.hashes], dtype=np.uint64)
        for i, hash1 in enumerate(self.hashes_binary):
            if i in self.duplicates:
                continue
            for j in range(i + 1, len(self.hashes_binary)):
                if j in self.duplicates:
                    continue
                hash2 = self.hashes_binary[j]
                if hamming(np.binary_repr(hash1), np.binary_repr(hash2)) < 0.25:  # adjust threshold as needed
                    self.duplicates.add(j)
        for i in sorted(self.duplicates, reverse=True):
            os.remove(self.image_paths[i])
            del self.image_paths[i]
            del self.images[i]
            del self.hashes[i]

    def remove_duplicates(self):
        """Remove similar images from the folder."""
        if os.path.isfile('hashes.csv'):
            with open('hashes.csv', mode='r') as file:
                reader = csv.reader(file)
                self.hashes = [hash_str for hash_str in next(reader)]
        else:
            self._load_images()
            self._calculate_hashes()
        self._find_duplicates()




if __name__ == '__main__':
    folder_path = 'D:\Andrew\Pictures\Grabber\\twitter_saver_2023'
    remover = ImageDuplicateRemover(folder_path)
    remover.remove_duplicates()

    print("D")
