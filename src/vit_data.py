
import sys
sys.path.append('.')


class ViTData(Dataset):

    def __init__(self, data_path, img_transform=transforms.Normalize((0.5), (0.5), (0.5)),
            channel_1_time=None):

        pickled = read_pickle(data_path)
        self.orfs = read_orfs_data('data/orfs_cd_paper_dataset.csv')

        (self.images, self.transcription, 
         self.orf_chroms, self.orf_names,
         self.orf_times) = pickled

        # If two images, set the channel 1 time as the first channel
        # for all remaining times
        self.load_data(channel_1_time=channel_1_time)

        self.channel_1_time = channel_1_time
        self.img_transform = img_transform
        self.transcription_unscaled = self.transcription
        self.transcription = scale(self.transcription.astype('float')).astype('float32')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = torch.tensor(self.images[idx])

        if self.img_transform is not None:
           data = self.img_transform(data)

        transcription = self.transcription[idx]
        orf_name = self.orf_names[idx]
        time = self.orf_times[idx]
        chrom = self.orf_chroms[idx]

        return data, transcription, orf_name, chrom, time

    def index_for(self, gene_name, time):
        orf_name = self.orfs[(self.orfs['name'] == gene_name) |
                             (self.orfs.index == gene_name)].index.values[0]

        index = np.arange(len(self))[(self.orf_names == orf_name) & 
                                       (self.orf_times == time)][0]

        return index

    def plot_index(self, idx):

        img, logfold_transcription, \
        orf_name, orf_chrom, time = self[idx]
        log_TPM = self.transcription_unscaled[idx]

        new_img_size = 256, 1024

        resized_img = np.moveaxis(img.numpy(), 0, 2)
        resized_img = cv2.resize(resized_img, (new_img_size[1], new_img_size[0]))

        print(resized_img.shape)

        plt.figure(figsize=(6, 4))
        plt.subplot(2, 1, 1)
        plt.imshow(resized_img[:, :, 1], vmin=-1.0, vmax=-0.0, cmap='Spectral_r', origin='lower', 
            extent=[-512, 512, 0, 256], aspect='auto')

        plt.suptitle(f"{self.orfs.loc[orf_name]['name']}, {time}' "
                     f"log-TPM: {log_TPM:.1f}")

    def load_data(self, channel_1_time):

        all_indices = np.arange(len(self.images))

        if channel_1_time == 'cac1_wt':
            wt_imgs, _, wt_orfs, wt_times = \
                read_pickle('data/cac1_pulse_chase/cac1_wt_mnase_40_32_128.pkl')
            images_1 = wt_imgs
            images_t = self.images
            time_t_indices = all_indices
        elif channel_1_time == 'dsb_wt_rep1':
            wt_imgs, _, wt_orfs, wt_times = \
                read_pickle('data/vit/dsb_wt_32x128.pkl')
            images_1 = wt_imgs
            images_t = self.images
            time_t_indices = all_indices
        # 1 channel, first channel is zeros (unused)
        elif channel_1_time == None:
            images_1 = None
            images_t = self.images
            time_t_indices = all_indices
        # Use one of the other time points as a channel
        else:
            time_1_indices = all_indices[self.orf_times == channel_1_time]
            time_t_indices = all_indices[self.orf_times != channel_1_time]

            images_1 = self.images[time_1_indices]
            images_t = self.images[time_t_indices]

        if images_1 is not None:
            # Duplicate channel 1 images so it matches remaining dataset
            repeat = images_t.shape[0]//images_1.shape[0]
            images_1 = np.repeat(images_1, repeat, axis=0)
            # Concatenate images to two channels, 
            self.images = np.concatenate([images_1, images_t], axis=1)
        else:
            self.images = images_t

        # reselect remaining data to match removing channel 1 data
        self.orf_chroms = self.orf_chroms[time_t_indices]
        self.transcription = self.transcription[time_t_indices]
        self.orf_times = self.orf_times[time_t_indices]
        self.orf_names = self.orf_names[time_t_indices]



if __name__ == '__main__':
    main()

