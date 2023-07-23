from utils import *
from operator import itemgetter
from itertools import groupby
import cv2
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--time_slices', default=5, type=int)

def generate_fixation_files(path, time_slices):
    print('Parsing fixations of ' + path + '...')
    filenames = [nm.split(".")[0] for nm in os.listdir(FIXATION_PATH + path)]

    def create_dirs(dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        dir_path = dir_path + '/' + path
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        return dir_path

    sal_vol_path = create_dirs(SAL_VOL_PATH + str(time_slices))
    fix_vol_path = create_dirs(FIX_VOL_PATH + str(time_slices))

    conv2D = GaussianBlur2D().cuda()

    print('Generating saliency volumes of ' + path + '...')
    for filename in tqdm(filenames):
        fixation_volume = parse_fixations([filename], FIXATION_PATH + path, progress_bar=False)[0]
        fix_timestamps = sorted([fixation for fix_timestamps in fixation_volume
                                        for fixation in fix_timestamps], key=lambda x: x[0])
        fix_timestamps = np.array([(min(int(ts * time_slices / TIMESPAN), time_slices-1), (x, y)) for (ts, (x, y)) in fix_timestamps])

        # Saving fixation map
        fix_vol = np.zeros(shape=(time_slices,H,W))
        for i, coords in fix_timestamps:
            fix_vol[i, coords[1] - 1, coords[0] - 1] = 1

        # Saving fixation list with timestamps
        compressed = np.array([(key, list(v[1] for v in valuesiter))
                            for key,valuesiter in groupby(fix_timestamps, key=itemgetter(0))])

        saliency_volume = get_saliency_volume(compressed, conv2D, time_slices)
        saliency_volume = saliency_volume.squeeze(0).squeeze(0).detach().cpu().numpy()

        for i, saliency_slice in enumerate(saliency_volume):
            cv2.imwrite(sal_vol_path + filename + '_' + str(i) + '.png', 255 * saliency_slice)
            cv2.imwrite(fix_vol_path + filename + '_' + str(i) + '.png', 255 * fix_vol[i])

args = parser.parse_args()
time_slices = args.time_slices
generate_fixation_files('train/', time_slices)
generate_fixation_files('val/', time_slices)
