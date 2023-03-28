# Gelbooru Downloader

This Python script allows you to download images from Gelbooru based on search strings. It can handle multiple search strings, and the number of images to download and the number of worker threads can be configured.

## Installation

1. Ensure you have Python 3.6+ installed.
2. Install the required dependencies:

``````python
pip install requests tqdm
``````



## Usage 

1. Import the `GelbooruDownloader` class from the script: 

```python 
from gelbooru_downloader import GelbooruDownloader
```



2. Create an instance of the `GelbooruDownloader` class with the desired parameters:

```python
search_strings = ["search_string1", "search_string2"]
output_dir = "output"
max_num = 1000
worker_num = 4

downloader = GelbooruDownloader(search_strings, output_dir, max_num, worker_num)
```

- `search_strings`: A list of search strings you want to download images for.
- `output_dir`: The output directory where the images will be saved (default: "output").
- `max_num`: The maximum number of images to download for each search string (default: 1000).
- `worker_num`: The number of worker threads to use for downloading images (default: 4).



3. Start the download process:

```python
downloader.download_all()
```

This will download images for each search string and save them in the output directory. Images will be saved in subdirectories named after the search strings, and the filenames will include the artist, date, post ID, and MD5 hash.



## Notes

- The script checks for existing files before downloading, and it will not redownload images if they already exist in the output directory.
- Progress bars are displayed for both fetching and downloading images, providing a visual representation of the progress for each search string.



## Example

Here's an example of using the GelbooruDownloader to download images with two different search strings:

```python
from gelbooru_downloader import GelbooruDownloader

search_strings = ["search_string1", "search_string2"]
output_dir = "output"
max_num = 1000
worker_num = 4

downloader = GelbooruDownloader(search_strings, output_dir, max_num, worker_num)
downloader.download_all()
```

This will download up to 1000 images for each search string using 4 worker threads and save them in the "output" directory. The images will be organized into subdirectories based on the search strings.



## License

This project is licensed under the GPL V3 License. See the [LICENSE](https://www.gnu.org/licenses/gpl-3.0.txt) file for details.



## Contributing

Please feel free to open an issue or submit a pull request with any improvements or bug fixes. All contributions are welcome!
