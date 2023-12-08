import os
from pathlib import Path

from fastai.data.transforms import get_image_files
from fastai.vision.utils import download_images, verify_images
from fastbook import search_images_bing

# from fastbook import *

key = os.environ.get('AZURE_SEARCH_KEY', '')


def download_imgs(base_path, labels, search_phrase):
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    for label in labels:
        dest = os.path.join(base_path, label)
        if not os.path.exists(dest):
            os.makedirs(dest)
        results = search_images_bing(key, f'{label} {search_phrase}')
        download_images(dest, urls=results.attrgot('contentUrl'))


def validate_images(img_root_path):
    fns = get_image_files(img_root_path)
    print(f'total images: {len(fns)}')
    failed = verify_images(fns)
    failed.map(Path.unlink)
    fns = get_image_files(img_root_path)
    print(f'total images after clean up: {len(fns)}')


if __name__ == '__main__':
    parrots_type = 'lovebird', 'green cheek conure', 'jako'
    base_path = './images'
    download_imgs(base_path, parrots_type, "parrot")
    validate_images(base_path)
