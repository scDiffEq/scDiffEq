
def _configure_optimization(param_groups, learning_rates, optimizers, schedulers, scheduler_kwargs):
    
    OptimizerDict, SchedulerDict = {}, {}

    for n, opt in enumerate(optimizers):
        OptimizerDict[n] = opt(param_groups[n], learning_rates[n])
        if schedulers[n]:
            SchedulerDict[n] = schedulers[n](OptimizerDict[n], **scheduler_kwargs[n])
        else:
            SchedulerDict[n] = None
            
    return list(OptimizerDict.values()), list(SchedulerDict.values())