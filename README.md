# chalbo-repo

To view the visualization: [Link](https://www.youtube.com/watch?v=ooIvjzcmfug)

This repository contains the work of the assignment given, details about how to use the scripts
and view the output are listed below. 

**Project Structure**: 
```bash
├── chalbo-task.pdf
├── experiments
│   ├── modularcnn.ipynb
│   ├── very_modularcnn.ipynb
│   ├── vgg_pretrained.ipynb
│   └── vgg_scratch.ipynb
├── main.py
├── models.py
├── out.mkv
├── README.md
├── requirements.txt
└── Task.pdf
```

`models.py`: Contains the code for the two neural networks `VerySmallCNN` and `SmallCNN`. 
`main.py`: Contains the inference script to actually run the code. It performs 
the data transformations, importing the datasets, and training as well as evalutation.

## To install the dependencies:

You can use the requirements.txt file and enter this command in your terminal. 
```bash
pip install -r requirements.txt
```

WARNING: Please use a virtual/conda environment for package isolation and just overall practice. 

## To run the model: 

Running the models is as simple as entering: `python main.py` in your terminal. If you would like to use the `SmallCNN()` instead, uncomment that line and make the necessary changes, wherever specified in the file.

In-depth details about the lines of code and their outputs are in `chalbo-task.pdf`.

## License: 

This code is not to be used publicly in any way. Done by Akshath Mangudi. 