<div align = "center">
<h3>
Computer vision for handwritten signature detection in pictures
</h3>
<img width = "200" src = /assets/readme_img/best-regards-pen.gif alt="Hand signature">
</div>

<p align="center">
  <a href="#the-project">Project</a> •
  <a href="#data-source-and-structure">Data source and structure</a> •
  <a href="#our-solutions">Solutions</a> •
  <a href="#how-to-use">How to use</a>
</p>

### The project

The aim of this (learning) project is to classify pictures of (administrative) documents as **signed vs. not signed**:
<div align = "center">
<img width = "200" src = /assets/readme_img/signed.tif alt="Signed doc">
<img width = "200" src = /assets/readme_img/unsigned.tif alt="Unsigned doc">
</div>
The document on the left is handwritten signed while the one on the right is not.

### Data source and structure

The images used to build this project are available [here]().
Please grab these assets and extract them in the root directory.
```
.
├── data
|   ├── test
|   └── train
|   └── train_xml
├── utils
├── .gitignore
├── main.py
└── README.md
```
### Solutions

#### 1. OpenCV manipulations 
This first approach consists in putting constraints on the conoutrs found in each document.
A handwritten signature should form a contour that is neither too small nor too large (relatively to the document size). Most of the signatures tend to take more space horizontally than vertically.
And the extent of the signature in its bounding box is smaller than typographed text.

We found that these constraints allow to reach a kappa score of 50%.

#### 2. Convolutional neural network (CNN) 
This second approach consists in ...

### How to use

You'll need [Python](https://www.python.org/) installed on your computer to clone and run this application.
From your command line:
```
# Clone this repository
$ git clone https://github.com/lyesds/signature_detection

# Go into the repository
$ cd signature_detection

# Install dependencies
$ pip install requirements.txt

# Run the main.py script
$ python run main.py
```


---
> GitHub [@lyesds](https://github.com/lyesds)