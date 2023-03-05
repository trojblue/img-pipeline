from concurrent.futures import ThreadPoolExecutor

from tqdm.auto import tqdm
import sdtools.imgops as iops
import sdtools.fileops as fops


def do_replace_info(src_dir, target_dir, add_suffix: bool, new_meta: str):
    """
    读取src_dir图片, 抹除水印后复制到target_dir
    :param src_dir:
    :param target_dir:
    :param add_suffix: 是否在文件里加入_cleaned后缀
    :param new_meta: 去除水印后填入的新水印
    :return:
    """
    fops.mkdir_if_not_exist(target_dir)
    image_files = fops.get_files_with_suffix(src_dir, fops.IMG_FILES)
    n_imgs = len(image_files)

    # 多线程, 并发
    # Process the images using ThreadPoolExecutor
    with tqdm(total=n_imgs, desc="remove info: ") as pbar:
        with ThreadPoolExecutor() as executor:
            # Submit mecha_tags task for each image file to be processed by mecha_tags thread
            tasks = [
                executor.submit(
                    iops.remove_watermark_single,
                    src_dir,
                    target_dir,
                    filename,
                    add_suffix,
                    new_meta,
                )
                for filename in image_files
            ]

            # Update the progress bar as tasks are completed
            for task in tasks:
                task.add_done_callback(lambda _: pbar.update())

            # Wait for all tasks to complete
            for task in tasks:
                task.result()


if __name__ == "__main__":
    src = "O:\===分拣===\==主题\98 fanbox\\NEW\__limited"
    img = "23.01.31_081416_DPM++ 2S a Karras_step35_cfg9.png"
    curr_img = "O:\===分拣===\==主题\98 fanbox\\NEW\__limited\\23.01.31_081416_DPM++ 2S a Karras_step35_cfg9.png"

    do_replace_info(src, src + "file", False, "updated")
