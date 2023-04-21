from dask.distributed import Client
import neg
import time
import dask_ml
client = Client('tcp://192.168.1.196:8786')
client.upload_file(r'C:\Users\10191\PycharmProjects\ModelsForC\neg.py')
from neg import abso
def square(x):
    return x ** 2


time0 = time.time()
A = client.map(square, range(10))
B = client.map(abso, A)
total = client.submit(sum, B)
total.result()
time1 = time.time()
print(time1 - time0)

time0 = time.time()
A = square(range(10))
B = neg(A)
total = sum(B)
time1 = time.time()
print(time1 - time0)
dask_ml.ensemble.RandomForestClassifier()