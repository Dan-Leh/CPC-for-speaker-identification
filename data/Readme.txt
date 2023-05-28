We used the LibriSpeech 'train-clean-100' dataset from http://www.openslr.org/12/.
The aligned phones for these examples were acquired with Kaldi and the data available from http://kaldi-asr.org/downloads/build/6/trunk/egs/librispeech/s5/.

The phone labels for the audio files are stored in aligned_phones.txt. There's a phone label every 10ms.

The train/test split enumerates all the training and test files. Please create your own validation set from the training set (or with cross-validation) and only eval on the test set for the final results.

The directory paths in LibriSpeech include the speaker-id, which allows to get the speaker label for every file.

The human readable text for every example can be found here: http://kaldi-asr.org/downloads/build/6/trunk/egs/librispeech/s5/data/train_clean_100/text.
