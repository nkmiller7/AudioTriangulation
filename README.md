# AudioTriangulation

## Set dataset path:

after downloading the dataset, open postprocess.py and update the base_folder_path variable to match the location where your dataset is stored.

## Run preprocessing:

execute postprocess.py and select option 3 to preprocess the raw data.
Once this step completes, run the script again and select option 4 to divide the dataset.

## Start training:

with the processed dataset ready, you can start training by running main.py.
If you're using Weights & Biases for logging, make sure to fill in your own WandB project and entity details (or disable WandB if you don't plan to use it).

## Make predictions:

once you have a trained model, you can predict the drone location from a single 8-channel audio file using predict.py.

### Basic usage:

```bash
python src/predict.py --audio path/to/your/audio.wav
```

### With a specific model:

```bash
python src/predict.py --audio data/Microphone_array/20241115_093128/output.wav --model saved_models/mel_96000_2048_1024_0_128_0_0/115_end_unet.pth
```

### Save results to JSON:

```bash
python src/predict.py --audio data/Microphone_array/20241115_093128/output.wav --output results.json
```

The script will output the drone's location in terms of:

- **Distance** from the microphone array (in meters)
- **Height** above ground (in meters)
- **Direction/Azimuth** - compass bearing to the drone (0-360°)
- **Facing** - direction the drone's camera is pointing (0-360°)
