try:
    from extensions.img_downloader.gelbooru_downloader import GelbooruDownloader
except ImportError:
    from gelbooru_downloader import GelbooruDownloader
def get_unique_strings(file1_path: str, file2_path: str) -> list:
    """Return a list of strings in file2 that are not in file1."""

    # Read the lines from the first file
    with open(file1_path, 'r') as file1:
        lines1 = set(line.strip() for line in file1)

    # Read the lines from the second file
    with open(file2_path, 'r') as file2:
        lines2 = set(line.strip() for line in file2)

    # Get the unique lines from the second file that are not in the first file
    unique_lines = list(lines2 - lines1)

    return unique_lines


def run_diff(file1_path: str, file2_path: str) -> None:
    """Print the unique strings in file2 that are not in file1.
    """

    unique_strings = get_unique_strings(file1_path, file2_path)
    downloader = GelbooruDownloader(unique_strings, output_dir="D:\Andrew\Pictures\Grabber\mai_saver")
    downloader.download_all()



if __name__ == '__main__':
    file1_path = '../../bin/my_following_dbr.txt'
    file2_path = '../../bin/mai_following_dbr.txt'
    run_diff(file1_path, file2_path)
