import torchvision.transforms.functional as TF


class PaddedResize:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        w, h = img.size
        if h > w:
            r = self.size / float(h)
            resized_w = int(w * r)
            resized_h = self.size
            img = img.resize((resized_w, resized_h))
            pad_x_left = int((self.size - resized_w) / 2.0)
            pad_x_right = self.size - resized_w - pad_x_left
            pad_y_top = 0
            pad_y_bottom = 0
        else:
            r = self.size / float(w)
            resized_w = self.size
            resized_h = int(h * r)
            img = img.resize((resized_w, resized_h))
            pad_x_left = 0
            pad_x_right = 0
            pad_y_top = int((self.size - resized_h) / 2.0)
            pad_y_bottom = self.size - resized_h - pad_y_top
        padding = (pad_x_left, pad_y_top, pad_x_right, pad_y_bottom)
        return TF.pad(img, padding=padding, padding_mode="symmetric")

    def __str__(self):
        return self.__class__.__name__ + f"(size={self.size})"
