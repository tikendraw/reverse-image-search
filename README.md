# Reverse Image Search Engine(Local)
This is a simple reverse image search tool that uses the Streamlit framework. It allows users to find similar images from a specified directory.

## Why is it helpful?
This tool can help identify similar images

## Installation
To use the reverse image search engine, simply follow these steps:

```bash

# clone the repo
git clone https://github.com/tikendraw/reverse-image-search.git

# go inside
cd reverse-image-search

# install with pip
pip install . 

```

## Run

bash command
```bash
# Run the app with command 
image_search

```
or just run the `launch.py` file
```bash
python launch.py
```
## Screenshot
<img src="./static/rev-image-search.jpg">

## How to use
*  Provide the directory path where the images are located. the create embeddings. 

* Upload an image.

* Number of similar Images to find. 

* Save Model (Optional): You can choose to save the model to speed up future searches if you plan to search in the same folder repeatedly. Enable the "Save Model" checkbox to save the model.

* Recursive Search (Optional): If you want to search for images recursively in child folders, enable the "Recursive" checkbox.

## Tech stack use
The reverse image search engine is built using the following technologies:

* Python
* transformers 
* pytorch 
* Pillow
* Chromadb(vector database)
* Streamlit
  
## How to contribute
Future updates:

- [ ] Facial Recognition
- [ ] Search Images with words

If you would like to contribute to the reverse image search engine, please feel free to open a pull request.

## License
The reverse image search engine is licensed under the MIT License.

I hope this is helpful!


