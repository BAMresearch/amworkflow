import psutil

threads_count = psutil.cpu_count()
threads = psutil.cpu_count(logical=False)
print(threads_count)
print(threads)