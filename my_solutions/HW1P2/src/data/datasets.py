import os
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


class AudioDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        root,
        phonemes,
        context=0,
        partition="train-clean-100",
        cepstral_normalization=False,
        partial_load=False,
    ):

        self.context = context
        self.phonemes = phonemes

        # Setting up the directories for mfcc and transcripts
        self.mfcc_dir = os.path.join(root, partition, "mfcc")
        self.transcript_dir = os.path.join(root, partition, "transcript")

        # Getting the list of mfcc and transcript files in sorted order to align them
        mfcc_names = sorted(os.listdir(self.mfcc_dir))
        transcript_names = sorted(os.listdir(self.transcript_dir))

        # Making sure that we have the same no. of mfcc and transcripts
        assert len(mfcc_names) == len(transcript_names)

        self.mfccs, self.transcripts = [], []
        scaler = StandardScaler()

        for i in range(len(mfcc_names)):
            mfcc_path = os.path.join(self.mfcc_dir, mfcc_names[i])
            transcript_path = os.path.join(self.transcript_dir, transcript_names[i])
            # Load a single mfcc
            mfcc = np.load(mfcc_path)  # Shape: T x 28, where T is the number of frames in the audio
            # Do Cepstral Normalization of mfcc
            if cepstral_normalization:
                mfcc = scaler.fit_transform(mfcc)
            # Load the corresponding transcript
            transcript = np.load(transcript_path)
            transcript = transcript[1:-1]  # Remove [SOS] and [EOS]
            # Append each mfcc to self.mfcc, transcript to self.transcript
            self.mfccs.append(mfcc)
            self.transcripts.append(transcript)

        # If partial_load is True, we only load the partial dataset
        if partial_load:
            self.mfccs = self.mfccs[: int(len(self.mfccs) * partial_load)]
            self.transcripts = self.transcripts[: int(len(self.transcripts) * partial_load)]
        # Concatenate all mfccs and transcripts
        self.mfccs = np.concatenate(self.mfccs, axis=0)
        self.transcripts = np.concatenate(self.transcripts, axis=0)

        # Length of the dataset is now the length of concatenated mfccs/transcripts
        self.length = len(self.mfccs)

        # Padding the mfccs to include context frames
        self.mfccs = np.pad(self.mfccs, ((self.context, self.context), (0, 0)), mode="constant", constant_values=0)

        # Converting phonemes in transcripts to indices
        self.transcripts = np.array([self.phonemes.index(phoneme) for phoneme in self.transcripts])

    def __len__(self):
        return self.length

    def __getitem__(self, ind):
        frames = self.mfccs[ind : ind + 2 * self.context + 1]
        # After slicing, you get an array of shape 2*context+1 x 28.
        # Move the flattening to the model to add data augmentation
        # frames = frames.flatten()
        frames = torch.FloatTensor(frames)  # Convert to tensors
        phonemes = torch.tensor(self.transcripts[ind])

        return frames, phonemes


class AudioTestDataset(torch.utils.data.Dataset):
    def __init__(self, root, phonemes, context=0, partition="test-clean", cepstral_normalization=False):
        self.context = context
        self.phonemes = phonemes

        # Setting up the directories for mfcc
        self.mfcc_dir = os.path.join(root, partition, "mfcc")

        # Getting the list of mfcc files in sorted order to align them
        mfcc_names = sorted(os.listdir(self.mfcc_dir))
        self.mfccs = []
        scaler = StandardScaler()

        for i in range(len(mfcc_names)):
            mfcc_path = os.path.join(self.mfcc_dir, mfcc_names[i])
            # Load a single mfcc
            mfcc = np.load(mfcc_path)  # Shape: T x 28, where T is the number of frames in the audio
            # Do Cepstral Normalization of mfcc
            if cepstral_normalization:
                mfcc = scaler.fit_transform(mfcc)

            # Append each mfcc to self.mfcc
            self.mfccs.append(mfcc)

        # Concatenate all mfccs
        self.mfccs = np.concatenate(self.mfccs, axis=0)

        # Length of the dataset is now the length of concatenated mfccs
        self.length = len(self.mfccs)

        # Padding the mfccs to include context frames
        self.mfccs = np.pad(self.mfccs, ((self.context, self.context), (0, 0)), mode="constant", constant_values=0)

    def __len__(self):
        return self.length

    def __getitem__(self, ind):

        frames = self.mfccs[ind : ind + 2 * self.context + 1]
        # After slicing, you get an array of shape 2*context+1 x 28.
        frames = torch.FloatTensor(frames)  # Convert to tensors

        return frames
