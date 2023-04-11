import tensorflow as tf
import os
# if tf.test.is_gpu_available(cuda_only=True):
if tf.test.is_built_with_cuda():
    print("TensorFlow can access CUDA")
else:
    print("TensorFlow cannot access CUDA")

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


path  = 'data/sample_download/mp3_download_all'
for i in os.listdir(path):
    print(i,len(os.listdir(os.path.join(path,i))))
