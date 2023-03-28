import time
import csv
import requests
from bs4 import BeautifulSoup
from typing import List, Tuple, Dict, Optional
from tqdm.auto import tqdm

class DanbooruArtistFinder:

    def __init__(self):
        self.base_url = "https://danbooru.donmai.us"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:97.0) Gecko/20100101 Firefox/97.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Pragma": "no-cache",
            "Cache-Control": "no-cache",
        }

    def _get_artist_tags(self, post_url: str) -> List[str]:
        response = requests.get(post_url)
        soup = BeautifulSoup(response.content, "html.parser")
        tags = soup.find_all("li", class_="tag-type-1")

        artists = []
        for tag in tags:
            artist_name = tag["data-tag-name"]
            artists.append(artist_name)

        return artists

    def flatten_list(self, nested_list):
        flattened_list = []
        for sublist in nested_list:
            for item in sublist:
                flattened_list.append(item)
        return flattened_list

    def find_artists(self, twitter_handles: List[str]) -> tuple[list[list[str]], list]:
        result = []
        notfound = []
        real_notfound = []
        pbar = tqdm(twitter_handles, desc="Finding artists")
        for handle in pbar:
            pbar.desc = f"Finding artists - {handle}"
            search_url = f"{self.base_url}/posts?tags=source%3Ahttps%3A%2F%2Ftwitter.com%2F{handle}"
            response = requests.get(search_url)
            soup = BeautifulSoup(response.content, "html.parser")
            post = soup.find("article", class_="post-preview")

            if post:
                post_id = post["data-id"]
                post_url = f"{self.base_url}/posts/{post_id}"
                artists = self._get_artist_tags(post_url)
                result.append(artists)
            else:
                notfound.append(handle)
            time.sleep(2)

        for handle in tqdm(notfound, desc="Finding artists - retries"):
            search_url = f"{self.base_url}/posts?tags=source%3Ahttps%3A%2F%2Ftwitter.com%2F{handle}"
            response = requests.get(search_url)
            soup = BeautifulSoup(response.content, "html.parser")
            post = soup.find("article", class_="post-preview")

            if post:
                post_id = post["data-id"]
                post_url = f"{self.base_url}/posts/{post_id}"
                artists = self._get_artist_tags(post_url)
                result.append(artists)
            else:
                real_notfound.append(handle)
                print(f"not found: {handle}")
            time.sleep(5)

        return self.flatten_list(result), notfound

def get_twitter_handles_from_csv(csv_file):

    with open(csv_file, "r", encoding='utf-8') as f:
        reader = csv.reader(f)
        reader.__next__()
        twitter_handles = [row[2] for row in reader]    # 'username'

    return twitter_handles

def run(csv_file):
    twitter_handles = get_twitter_handles_from_csv(csv_file)
    finder = DanbooruArtistFinder()
    artists, notfound = finder.find_artists(twitter_handles)  # iumu, dino_(dinoartforame), tota_(sizukurubiks)
    for i in artists:
        print(i)
    print(notfound)

def debug():
    post_url = "https://danbooru.donmai.us/posts/6081028"
    finder = DanbooruArtistFinder()
    artists = finder._get_artist_tags(post_url)
    print(artists)

if __name__ == "__main__":
    # csv_file = "../../bin/trojblue_following.csv"
    csv_file = "../../bin/maimuro_following.csv"
    run(csv_file)
    # run()

