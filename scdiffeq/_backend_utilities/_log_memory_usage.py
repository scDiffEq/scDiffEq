import psutil, os

def log_memory_usage(epoch):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 ** 2  # in MB
    print(f"Epoch {epoch} - Memory Usage: {mem:.2f} MB")
