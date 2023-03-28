import os
import glob
import requests
from urllib.parse import quote
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


class GelbooruDownloader:
    def __init__(self, search_strings, output_dir="output", max_num=1200, worker_num=4):
        """
        :param search_strings: list of strings to search for
        :param output_dir: directory to save images to
        :param max_num: maximum number of images to download PER ARTIST
        :param worker_num: number of workers to use for downloading
        """
        self.search_strings = search_strings
        self.output_dir = output_dir
        self.max_num = max_num
        self.worker_num = worker_num

    def _create_output_path(self, artist, date, post_id, md5, ext):
        artist_dir = os.path.join(self.output_dir, artist)
        os.makedirs(artist_dir, exist_ok=True)
        filename = f"{artist}__{date.strftime('%Y%m%d')}__{post_id}__{md5}.{ext}"
        return os.path.join(artist_dir, filename)

    def _check_existing_file(self, artist, date, post_id, md5):
        artist_dir = os.path.join(self.output_dir, artist)
        os.makedirs(artist_dir, exist_ok=True)
        search_pattern = f"{artist}__{date.strftime('%Y%m%d')}__{post_id}__{md5}.*"
        existing_files = glob.glob(os.path.join(artist_dir, search_pattern))
        return existing_files[0] if existing_files else None

    def _download_image(self, post):
        artist = post["search_string"]
        date = datetime.strptime(post["created_at"], "%a %b %d %H:%M:%S %z %Y")
        post_id = post["id"]
        md5 = post["md5"]
        file_url = post["file_url"]

        existing_file = self._check_existing_file(artist, date, post_id, md5)
        if existing_file:
            print(f"File already exists: {existing_file}")
            return None

        response = requests.get(file_url, stream=True)
        if response.status_code == 200:
            content_type = response.headers.get("Content-Type")
            ext = content_type.split("/")[-1] if content_type else "jpg"
            output_path = self._create_output_path(artist, date, post_id, md5, ext)

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return output_path
        return None

    def _fetch_posts(self, search_string, page):
        url = f"https://gelbooru.com/index.php?page=dapi&s=post&q=index&tags={quote(search_string)}&pid={page}&json=1"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        return None

    def _fetch_all_posts(self, search_string):
        all_posts = []
        page = 0

        with tqdm(
            total=self.max_num, desc=f"Fetching images for '{search_string}'"
        ) as progress_bar:
            while len(all_posts) < self.max_num:
                result = self._fetch_posts(search_string, page)
                if not result:
                    break

                posts = result.get("post", [])
                if not posts:
                    break

                for post in posts:
                    post["search_string"] = search_string
                    all_posts.append(post)

                    progress_bar.update(1)
                    if len(all_posts) >= self.max_num:
                        break

                page += 1

        return all_posts

    def _download_all_posts(self, all_posts):
        with ThreadPoolExecutor(max_workers=self.worker_num) as executor:
            futures = [
                executor.submit(self._download_image, post) for post in all_posts
            ]

            with tqdm(total=len(all_posts), desc="Downloading images") as progress_bar:
                for future in as_completed(futures):
                    progress_bar.update(1)

    def download_all(self):
        for search_string in self.search_strings:
            all_posts = self._fetch_all_posts(search_string)
            self._download_all_posts(all_posts)




if __name__ == "__main__":
    search_strings = ["mittye97", "uppi", "happybiirthd"]
    downloader = GelbooruDownloader(search_strings)
    downloader.download_all()
