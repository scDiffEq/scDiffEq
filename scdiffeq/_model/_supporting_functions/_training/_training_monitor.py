import time

class SmoothedAverage(object):

    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.value = None
        self.smoothed_avg = 0

    def update(self, value):
        if self.value is None:
            self.smoothed_average = value
        else:
            self.smoothed_average = _calculate_smoothed_average(self.smoothed_average, self.momentum, value)
        self.value = value

def _calculate_smoothed_average(average, momentum, new_value):
    
    smoothed_average = average * momentum + new_value * (1 - momentum)
    
    return smoothed_average

class _TrainingMonitor(object):
    def __init__(self, momentum=0.99):
        
        self.current_epoch = 0
        self.train_loss = []
        self.valid_loss = []
        self.smoothed_train_loss = SmoothedAverage(momentum)
        self.smoothed_valid_loss = SmoothedAverage(momentum)
    
    def reset(self, reset_smoothed_average=True):
        self.train_loss = []
        self.valid_loss = []
        self.start_time = time.time()
        
        if reset_smoothed_average:
            self.smoothed_train_loss.reset()
            self.smoothed_valid_loss.reset()
        
    def update_loss(self, loss, validation=False):

        """
        If validation indicated, validation loss is updated. Otherwise,
        training loss is updated.

        Parameters:
        -----------
        loss
            type: float

        validation
            default: False
            type: bool
        """
                
        if validation:
            self.valid_loss.append(loss)
            self.smoothed_valid_loss.update(loss)
        else:
            self.train_loss.append(loss)
            self.smoothed_train_loss.update(loss)
            
    def start_timer(self):
        
        try: 
            self.backup_t0 = self.start_time
            self.start_time = time.time()
        except:
            self.start_time = time.time()            
        
    def update_time(self, round_to=3):

        """
        Update time tracking.

        Parameters:
        -----------
        round_to
            The number of decimals to be rounded to in keeping track of time. 
            default: 3
            type: int

        Returns:
        --------
        None
            self.current_time and self.elapsed_time are updated.
        """

        self.current_time = time.time()
        self.elapsed_time = round(self.current_time - self.start_time, round_to)