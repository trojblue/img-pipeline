import os
import re
import requests
from datetime import datetime
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed
import glob

class GelbooruDownloader:
    def __init__(self, search_strings, output_dir='output', max_num=1000, worker_num=4):
        self.search_strings = search_strings
        self.output_dir = output_dir
        self.max_num = max_num
        self.worker_num = worker_num

    def _create_output_path(self, artist, date, post_id, md5, ext):
        artist_dir = os.path.join(self.output_dir, artist)
        os.makedirs(artist_dir, exist_ok=True)
        filename = f"{artist}__{date.strftime('%Y%m%d')}__{post_id}__{md5}.{ext}"
        return os.path.join(artist_dir, filename)

    def _download_image(self, post):

        artist = post['search_string']
        date = datetime.strptime(post['created_at'], '%a %b %d %H:%M:%S %z %Y')
        post_id = post['id']
        md5 = post['md5']
        file_url = post['file_url']

        # Check if a file already exists, ignoring the extension
        artist_dir = os.path.join(self.output_dir, artist)
        os.makedirs(artist_dir, exist_ok=True)
        search_pattern = f"{artist}__{date.strftime('%Y%m%d')}__{post_id}__{md5}.*"
        existing_files = glob.glob(os.path.join(artist_dir, search_pattern))
        if existing_files:
            print(f"File already exists: {existing_files[0]}")
            return None

        response = requests.get(file_url, stream=True)
        if response.status_code == 200:
            content_type = response.headers.get('Content-Type')
            ext = content_type.split('/')[-1] if content_type else 'jpg'

            output_path = self._create_output_path(artist, date, post_id, md5, ext)

            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return output_path
        return None

    def _fetch_posts(self, search_string, page):
        url = f"https://gelbooru.com/index.php?page=dapi&s=post&q=index&tags={quote(search_string)}&pid={page}&json=1"
        response = requests.get(url)
        if response.status_code == 200:
            resp = response.json()
            return resp
        return None

    def _download_posts(self, search_string):
        page = 0
        downloaded_count = 0
        with ThreadPoolExecutor(max_workers=self.worker_num) as executor:
            while downloaded_count < self.max_num:
                result = self._fetch_posts(search_string, page)
                if not result:
                    break

                posts = result.get("post", [])

                # Break the loop if there are no more posts
                if not posts:
                    break

                for post in posts:
                    post['search_string'] = search_string

                futures = [executor.submit(self._download_image, post) for post in posts]
                for future in as_completed(futures):
                    downloaded_count += 1
                    if downloaded_count >= self.max_num:
                        break

                page += 1


    def download_all(self):
        for search_string in self.search_strings:
            print(f"Downloading images for '{search_string}'")
            self._download_posts(search_string)


if __name__ == '__main__':
    search_strings = ["mittye97", "uppi", "happybiirthd"]
    downloader = GelbooruDownloader(search_strings)
    downloader.download_all()
