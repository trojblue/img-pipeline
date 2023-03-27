import requests
from bs4 import BeautifulSoup
from typing import List
from tqdm.auto import tqdm

class DanbooruArtistFinder:

    def __init__(self):
        self.base_url = "https://danbooru.donmai.us"

    def _get_artist_tags(self, post_url: str) -> List[str]:
        response = requests.get(post_url)
        soup = BeautifulSoup(response.content, "html.parser")
        tags = soup.find_all("li", class_="tag-type-1")

        artists = []
        for tag in tags:
            artist_name = tag["data-tag-name"]
            artists.append(artist_name)

        return artists

    def find_artists(self, twitter_handles: List[str]) -> List[List[str]]:
        result = []
        for handle in tqdm(twitter_handles, desc="Finding artists"):
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
                result.append([])

        return result


def run():
    twitter_handles = ["iumukam", "Dino_illus", "kakikakiken"]
    finder = DanbooruArtistFinder()
    artists = finder.find_artists(twitter_handles)  # iumu, dino_(dinoartforame), tota_(sizukurubiks)
    print(artists)

def debug():
    post_url = "https://danbooru.donmai.us/posts/6081028"
    finder = DanbooruArtistFinder()
    artists = finder._get_artist_tags(post_url)
    print(artists)

if __name__ == "__main__":
    run()
