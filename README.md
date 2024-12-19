# Tutorial
## Installation and Environment Setup
Using Python 3.12.4
#### First, create the environement  
`python -m venv .venv`
#### Then activate the environment. This will depend on OS/platform. Here is the command for Windows:
`.venv\Scripts\activate`
#### Install the package requirements with pip:
`pip install -r requirements.txt`
  
## Usage Example
View demo.ipynb for a usage example

## Troubleshooting Guide
- If getting stuck in `get_collapsed_words()`, consider increasing the `search_reduction_factor` parameter. It defaults to 25, but depending on the memory on your machine, this may be too low.
- If getting the distance exception in `get_collapsed_words()`, try decreasing the collapse factor or increasing the `search_reduction_factor` parameter.
- If there appears to be no difference between collapse factors, make sure you have either deleted `token_to_new_id` and `id_to_new_id`, or set the `use_saved` parameter to false in `get_collapsed_words()`.
- If you are getting nonsense results, ensure that you are passing the train dataset's `id_to_new_id` to `create_dataset()` when creating the validation dataset.
- If having issues with the collapse factor, make sure you are using the inverse of the percent collapse. So to reduce the language by 30%, set the collapse factor to `1/0.3`.
- If missing the `base_tokenizer` directory, use something like the following, extract the snapshot directory, and rename it to `base_tokenizer`:  
    - `from huggingface_hub import snapshot_download`  
    - `a = snapshot_download(repo_id="roneneldan/TinyStories-33M")`
-