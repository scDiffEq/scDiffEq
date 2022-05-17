
def _test_bool(epoch, test_freq):
    
    if test_freq:
        if epoch % test_freq == 0:
            return True
        else:
            return False
    else:
        return False