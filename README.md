# AudioTriangulation

## Set dataset path:
after downloading the dataset, open postprocess.py and update the base_folder_path variable to match the location where your dataset is stored.


## Run preprocessing: 
execute postprocess.py and select option 3 to preprocess the raw data.
Once this step completes, run the script again and select option 4 to divide the dataset.


## Start training:
with the processed dataset ready, you can start training by running main.py.
If you're using Weights & Biases for logging, make sure to fill in your own WandB project and entity details (or disable WandB if you don't plan to use it).