For your final project, you will implement a small computer vision system that uses (at least) two of the methods/tasks we learned about in class to solve a higher-level computer-vision problem. See below for a specific suggestion for a project. You are welcome to propose your own project. If you'd like to do this, please discuss with the TAs whether or not it is a good fit (of sufficient complexity, feasible given the available resources, etc.).

To help you get started, we've created a Google Colab notebook which showcases how to use some models from huggingfaceLinks, a fantastic resource for running a wide variety of modern computer vision methods: https://colab.research.google.com/drive/1kBUjjWEHN5OxNrQUMqr-L4rDjYdVLk2n?usp=sharingLinks to an external site..

If you aren't feeling particularly creative, here's a suggested project:

Create a mock-up of an in-dash Bird's Eye View (BEV) system (here's a descriptionLinks to an external site. of what those can do) for a car. This is a feature that is being integrated with many new vehicles to support autonomous driving, assisted parking, or situational awareness for the driver.
Existing BEV systems are constructed from a set of fixed cameras with known relative geometry. For this project, you would do this with just a single camera taking pictures in many different directions and then estimate the geometric relationship between them (e.g., using panorama stitching).
It would be ideal if you not only showed a view of the scene from a top-down perspective, but you also showed the location of selected objects in the overhead view. For example, you could highlight the location of pedestrians and cars using object-specific icons.
You can assume that the world is planar, or you could leverage a monocular depth estimation model to give more accurate depth estimates.
To support this, you should collect a few different sets of images in varied locations and then automatically process them to generate an informative overhead map.
There are many different ways to approach this problem. You are welcome to choose the way you think makes the most sense, or is just more interesting for you. Hopefully that gives you a sense of the type of project we are looking for.

Notes:

You may work in groups of 1 to 4 (although 2-3 is probably best). The complexity/quality should scale commensurately with group size.
Requirements:

By the project deadline:
You must submit your source code and data (assuming it's not too big) as a zip file with sufficient information to easily run your code. We recommend you implement your project as a self-contained Juypter/Colab notebook or a set of Juypter/Colab notebooks.
You should include a readme that describes the overall scope of the project, or have a sufficiently descriptive header in your notebook.
Your final system should run on one example dataset with minimal user intervention.
You are strongly encouraged to generate intermediate visualizations to help you (and the TAs) understand how the model is working and where it is failing.
Notes:

You are welcome to use Generative AI systems during development, just be sure to document exactly what you used and how you used it.
Not Required:

You do not need to train a model. You are welcome to combine existing pre-trained models.
