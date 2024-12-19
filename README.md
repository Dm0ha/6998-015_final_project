# Tutorial
## Installation and Environment Setup
Using Python 3.12.4
#### First, create the environement  
`python -m venv .venv`
#### Then activate the environment. This will depend on OS/platform. Here is the command for Windows:
`.venv\Scripts\activate`
#### Install torch
`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
#### Install the other package requirements with pip:
`pip install -r requirements.txt`  

  
## Usage Examples
We have two usage examples/demonstration. 
- The first is in `demo_experiment.ipynb`, which showcases an experiment like we conducted in our research to determine the efficacy of the language collapse methodology. 
- The second is in `demo_usage.ipynb`, which is similar but showcases what a researcher could do using the method. Rather than testing the accuracy of the method, it demonstrates how the method could be used.

## Troubleshooting Guide
- If getting stuck in `get_collapsed_words()`, consider increasing the `search_reduction_factor` parameter. It defaults to 25, but depending on the memory on your machine, this may be too low.
- If getting the distance exception in `get_collapsed_words()`, try decreasing the collapse factor or increasing the `search_reduction_factor` parameter.
- If there appears to be no difference between collapse factors, make sure you have either deleted `token_to_new_id` and `id_to_new_id`, or set the `use_saved` parameter to `False` in `get_collapsed_words()`.
- If you are getting nonsense results for validation tests, ensure that you are passing the train dataset's `id_to_new_id` to `create_dataset()` when creating the validation dataset.
- If having issues related to methods using the collapse factor, make sure you are using the inverse of the percent collapse. In other words, to reduce the language by 30%, set the collapse factor to `1/0.3`.
- If missing the `base_tokenizer` directory, error handling should catch it and automatically set it up. If it does not work as expected, run the following lines of code extract the snapshot directory, and rename it to `base_tokenizer`:  
    - `from huggingface_hub import snapshot_download`  
    - `snapshot_download(repo_id=id, cache_dir="./all_tokenizer_files")`
- If getting an error when trying to preprocess the dataset for the MLM task, ensure that you are passing a 2D array to `pad_and_mask()`, and the array it returns for `add_mlm_masking()`.
- For the `train()` function, ensure that the model used is an instance of `BertForMaskedLM`.
- If you are getting file not found errors when running tests manually, make sure that you have run the collapse sequence first in `generator.py`, and are referencing the proper dataset names that it generates.
- If you get a file not found error related to models/losses when running tests manually, ensure you have created the directory specified in the file, e.g. `experiment_data/collapse/losses/`.