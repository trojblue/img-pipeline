from typing import List
from PIL import Image, ImageDraw, ImageFont

class GridMaker:
    def __init__(self, binary_imgs: List[Image.Image], xy: str, xy_len: int, tags: List[str]):
        self.binary_imgs = binary_imgs
        self.xy = xy
        self.xy_len = xy_len
        self.tags = tags

        # set default values for variables
        self.padding = 10
        self.tag_padding = 20
        self.background_color = (33, 33, 33)
        self.tag_color = (200, 200, 200)
        self.font_path = "/path/to/font.ttf"

        # calculate image dimensions and font size
        self.width, self.height = self.binary_imgs[0].size
        self.final_width, self.final_height = self.get_wh()
        self.font_size = int(0.03 * self.height)

    def get_wh(self):
        if self.xy == 'x':
            rows = self.xy_len
            cols = len(self.binary_imgs) // rows
        elif self.xy == 'y':
            cols = self.xy_len
            rows = len(self.binary_imgs) // cols
        else:
            rows = int(len(self.binary_imgs) ** 0.5)
            cols = (len(self.binary_imgs) + rows - 1) // rows

        return cols * (self.width + self.padding) + self.padding, rows * (self.height + self.padding) + self.tag_padding + self.padding

    def create_background(self):
        background = Image.new('RGB', (self.final_width, self.final_height), self.background_color)
        draw = ImageDraw.Draw(background)
        return background, draw

    def add_images(self, background, draw):
        x_offset = self.padding
        y_offset = self.tag_padding
        for i, img in enumerate(self.binary_imgs):
            col = i % (self.final_width // self.width)
            row = i // (self.final_width // self.width)
            pos = (x_offset + col * (self.width + self.padding), y_offset + row * (self.height + self.padding))
            background.paste(img, pos)

        return background

    def add_tags(self, background, draw):
        font = ImageFont.truetype(self.font_path, self.font_size)
        x_offset = self.padding
        y_offset = 0
        if self.xy == 'x':
            y_offset = self.final_height - self.tag_padding + self.padding
            col_width = (self.final_width - 2 * self.padding) // self.xy_len
            for i, tag in enumerate(self.tags):
                tag_w, tag_h = draw.textsize(tag, font=font)
                tag_x = x_offset + (col_width * i) + ((col_width - tag_w) // 2)
                tag_y = self.final_height - self.tag_padding + self.padding + ((self.tag_padding - tag_h) // 2)
                draw.text((tag_x, tag_y), tag, font=font, fill=self.tag_color)
        else:
            x_offset = self.final_width - self.tag_padding + self.padding
            row_height = (self.final_height - 2 * self.padding) // self.xy_len
            for i, tag in enumerate(self.tags):
                tag_w, tag_h = draw.textsize(tag, font=font)
                tag_x = self.final_width - self.tag_padding + self.padding + ((self.tag_padding - tag_w) // 2)
                tag_y = y_offset + (row_height * i) + ((row_height - tag_h) // 2)
                draw.text((tag_x, tag_y), tag, font=font, fill=self.tag_color)

        return background




if __name__ == '__main__':
    # list of binary images

    binary_imgs = [Image.open("image1.png"), Image.open("image2.png"), Image.open("image3.png")]

    # list of tags corresponding to the images
    tags = ["tag1", "tag2", "tag3"]

    # set the number of columns or rows for the grid
    xy = "auto"
    xy_len = 0

    # create the GridMaker object
    grid = GridMaker(binary_imgs, xy, xy_len, tags)

    # create the grid image
    grid_image = grid.concat_images()

    # save the grid image
    grid_image.save("grid_image.png")