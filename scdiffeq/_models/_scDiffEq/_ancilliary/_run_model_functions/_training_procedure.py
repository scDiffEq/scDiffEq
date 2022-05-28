

# import local dependencies #
# ------------------------- #
from ._progress_bar import _progress_bar


def _training_procedure(model, X, t):
    
    Learner = model._Learner
    ModelManager = model._ModelManager
    TrainingProgram = model._TrainingProgram
    
    progress_bar = _progress_bar(TrainingProgram, Learner._training_epoch_count)
    
    for epoch in progress_bar:
        print("epoch: {}".format(epoch))
        Learner.pass_train(X, t)
        
#         if epoch % TrainingProgram["validation_frequency"]:
#             Learner.validate()
        