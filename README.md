# TNSFnets
Tensor Navier-Stokes Flow nets - Physics Informed Neural Networks

Some of the codes need this experiments to run: https://drive.google.com/drive/folders/1Kl9U2Q1BvAQAaP5W3w6HpkXs79gT7TP6?usp=sharing

Folder structures:

TNSFnets
  -> cube
  
  ->-> data
  
  ->->-> cube_00
  
  ->->->-> files .npy to train and test the model
  
  ->->->-> folder of the experiments slices, the exact ones ps.: to verify the model: "y_equal_15_exact", and the predict ones: "y_equal_15_pred"
  
  ->->-> cube_01
  
  ->->->-> files .npy to train and test the model
  
  ->->->-> folder of the experiments slices, the exact ones ps.: to verify the model: "y_equal_15_exact", and the predict ones: "y_equal_15_pred"
  
  ->-> models
  
  ->-> (files like cube_data, cube_plotting, cube_test, etc)
  
  -> figures
  
  ->-> experiments folders ("beltrami_3d", "cube_00", "cube_01", etc)
  
  ->->-> folders of the experiments slices, like: "y_equal_15_exact", "z_equal_15_exact", "z_equal_10_exact"
  
  ->->-> images of the losses, and other images of the training proccess
  
  -> models
  
  -> sims
  
  ->-> here is the folder avaible to download in google drive, like: "cube_00" or "cube_01"
  
  ->->-> timesteps
  
  ->->->-> files of the exact positions (x, y, z) and velocities (u, v, w)
  
  -> (files like .gitignore, .git, BeltramiFlow..., CylinderWake... etc)
  
  
  
  To run the experiments:
  
  1- execute the experiment_data.py file, here you can define wich slice of the experiment you're going to use in training and others parameters
  
  2- training the model (experiment_training.py), here you can define the model name and other training parameters;
  
  3- test the model (experiment_test.py), here the predict files will be generated in the folder you especified/created;
  
  4- plot (experiment_plotting.py), self explanatory too.
  
  ps.: there is the networks python files ("NSFnet_fluidborders_model.py", etc), you can customize then too and specify in the import of the training file which network you're going to use.
