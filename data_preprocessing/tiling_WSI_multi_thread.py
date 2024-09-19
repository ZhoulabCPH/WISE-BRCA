import os
import concurrent.futures

from histolab.slide import Slide
from histolab.tiler import GridTiler


def tiling_WSI_BC(slide_path, output_path):
    base_Mpp = 0.488
    patch_size = 224
    slide = Slide(slide_path, output_path, use_largeimage=True)
    mpp = float(slide.properties['aperio.MPP'])
    trg_patch_size = (patch_size * base_Mpp) / mpp
    trg_patch_size = round(trg_patch_size)
    grid_tiles_extractor = GridTiler(
        tile_size=(trg_patch_size, trg_patch_size),
        level=0,
        check_tissue=True,
        tissue_percent=50,
        pixel_overlap=0,
        prefix="",
        suffix=".png",
    )
    grid_tiles_extractor.extract(slide)


def process_slide(slide_name):
    slide_path = rf'{slides_path}/{slide_name}'
    output_path = rf'{patches_path}/{slide_name}/'

    if not os.path.exists(output_path) or os.listdir(output_path) == []:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print(f'Tiling {slide_name}.')
        tiling_WSI_BC(slide_path, output_path)
        print(f'{slide_name} finish!')


if __name__ == '__main__':
    slides_path = '../datasets/WSIs'
    patches_path = '../datasets/Patches_224'
    slides = list(os.listdir(rf'{slides_path}'))
    slides = list(set([slide for slide in slides if slide not in os.listdir(rf'{patches_path}')]))

    num_threads = 4
    with concurrent.futures.ThreadPoolExecutor(num_threads) as executor:
        executor.map(process_slide, slides)
