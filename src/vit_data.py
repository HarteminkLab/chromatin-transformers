
import sys
sys.path.append('.')


import pandas as pd
import numpy as np
from src.utils import read_pickle


def load_rna_data():
    TPM_path = 'data/vit/cd_rna_seq_TPM.csv'
    TPM = pd.read_csv(TPM_path).set_index("orf_name")
    return TPM


def load_cd_img_data():

    pickle_paths = ('data/vit/vit_imgs_DM498_MNase_rep1_0_min.pkl',
                    'data/vit/vit_imgs_DM499_MNase_rep1_7.5_min.pkl',
                    'data/vit/vit_imgs_DM500_MNase_rep1_15_min.pkl',
                    'data/vit/vit_imgs_DM501_MNase_rep1_30_min.pkl',
                    'data/vit/vit_imgs_DM502_MNase_rep1_60_min.pkl',
                    'data/vit/vit_imgs_DM503_MNase_rep1_120_min.pkl',

                    'data/vit/vit_imgs_DM504_MNase_rep2_0_min.pkl',
                    'data/vit/vit_imgs_DM505_MNase_rep2_7.5_min.pkl',
                    'data/vit/vit_imgs_DM506_MNase_rep2_15_min.pkl',
                    'data/vit/vit_imgs_DM507_MNase_rep2_30_min.pkl',
                    'data/vit/vit_imgs_DM508_MNase_rep2_60_min.pkl',
                    'data/vit/vit_imgs_DM509_MNase_rep2_120_min.pkl')

    i = 0
    df = pd.DataFrame()
    times = np.array([])
    orfs = np.array([])

    all_imgs = None
    for path in pickle_paths:        
        filesplit = path.split('/')[-1].split('_')[2:]
        dm, rep, time = filesplit[0], filesplit[2], filesplit[3]
        desc, imgs = read_pickle(path)

        if all_imgs is None: all_imgs = imgs
        else:
            all_imgs = np.concatenate([all_imgs, imgs])
        
        times = np.append(times, np.repeat(float(time), len(imgs)))
        orfs = np.append(orfs, np.array(desc['orfs']))
        
        df.loc[i, 'DM'] = dm
        df.loc[i, 'replicate'] = rep
        df.loc[i, 'time'] = float(time)
        df.loc[i, 'path'] = path
        i += 1

    return df, all_imgs, orfs, times


if __name__ == '__main__':
    main()

