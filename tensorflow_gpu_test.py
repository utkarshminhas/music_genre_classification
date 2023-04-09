import tensorflow as tf

if tf.test.is_gpu_available(cuda_only=True):
    print("TensorFlow can access CUDA")
else:
    print("TensorFlow cannot access CUDA")

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

