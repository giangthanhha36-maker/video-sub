import numpy as np
from PIL import Image


def load_img(img_p: str):
    """
    将图片文件加载为图像数组，确保图像模式为RGB。

    参数:
    img_p (str): 图片文件的路径。

    返回:
    Image对象: 加载的图像，确保其模式为RGB。
    """
    # [GIU NGUYEN - DA COMMENT] Cach cu: Image.open mo anh kieu "lazy" va GIU file handle mo.
    # Khi xu ly hang nghin frame (extract_mask) se ro ri file descriptor -> loi
    # "OSError: [Errno 24] Too many open files".
    # img = Image.open(img_p)
    # if img.mode == "RGBA":
    #     img = img.convert("RGB")
    # return img

    # [THAY DOI] Mo anh trong khoi `with` de DONG file ngay sau khi doc xong.
    #   - convert("RGB"): luon tra ve anh RGB (giu nguyen hanh vi dau ra RGB).
    #   - .copy(): nap du lieu pixel ra bo nho roi tach khoi file -> dong duoc file,
    #     khong con ro ri file handle.
    # VAN tra ve doi tuong PIL.Image (KHONG doi sang numpy) de giu tuong thich
    # voi cac cho dung API PIL nhu image.size, .convert, .crop, .resize...
    with Image.open(img_p) as img:
        return img.convert("RGB").copy()


def load_img_to_array(img_p: str):
    """
    将图片文件加载为numpy数组格式。

    参数:
    img_p: str
        图片文件的路径。

    返回:
    numpy.ndarray
        表示图片的numpy数组
    """
    return np.array(load_img(img_p))


def save_array_to_img(img_arr: np.ndarray, img_p: str):
    """
    将图像的数组表示形式保存为图片文件。

    参数:
    img_arr: numpy.ndarray, 图像的数组表示，其中每个元素是一个像素值。
    img_p: str, 要保存的图片文件的路径，包括文件名和扩展名。

    返回:
    无返回值。该函数将图像数组保存为指定路径的图片文件。
    """
    Image.fromarray(img_arr.astype(np.uint8)).save(img_p)
