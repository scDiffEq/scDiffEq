class _Messages_:
    
    def __init__(self):
        
        """Class of functions to organize and communicate messages through the scDiffEq module."""
        
    def single_trajectory(self):
        
        self.single_trajectory_msg = "Only a single trajectory present in the dataset. Data is to be split by trajectories. Thus, the validation and test groups will be null and this trajectory will be attributed to the training set and doubly to the test set as it is assumed that overfitting / testing is the goal."
        
        print(self.single_trajectory_msg, end='\n\n')
        
    def overfit(self):
        
        self.overfit_msg = "Overfit mode employed. Data will not be used for validation."
        
        print(self.overfit_msg, end='\n\n')