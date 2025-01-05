# Put the wav file in dataset_raw and run the script

python resample.py
python preprocess_flist_config.py --speech_encoder vec768l12 --vol_aug
python preprocess_hubert_f0.py --f0_predictor dio 