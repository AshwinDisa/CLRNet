import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import torch
import time

class TRTInference:
    def __init__(self, engine_path, postprocess=None):
        self.logger = trt.Logger(trt.Logger.WARNING)

        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Identify input/output tensor names
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)

        # Static batch=1 engine → shapes have no -1
        self.input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        self.output_shape = tuple(self.engine.get_tensor_shape(self.output_name))

        self.input_dtype = trt.nptype(self.engine.get_tensor_dtype(self.input_name))
        self.output_dtype = trt.nptype(self.engine.get_tensor_dtype(self.output_name))

        # Allocate host and device buffers
        n_input = int(np.prod(self.input_shape))
        n_output = int(np.prod(self.output_shape))

        self.input_host = np.empty(n_input, dtype=self.input_dtype)
        self.output_host = np.empty(n_output, dtype=self.output_dtype)

        self.input_device = cuda.mem_alloc(self.input_host.nbytes)
        self.output_device = cuda.mem_alloc(self.output_host.nbytes)

        self.stream = cuda.Stream()
        self.postprocess = postprocess

    def infer(self, input_image: np.ndarray):
        assert input_image.shape == self.input_shape, f"Input must be {self.input_shape}"
        assert input_image.dtype == self.input_dtype

        # Flatten and copy to pagelocked buffer
        np.copyto(self.input_host, input_image.ravel())

        # Async transfer to device
        cuda.memcpy_htod_async(self.input_device, self.input_host, self.stream)

        # Set tensor addresses
        self.context.set_tensor_address(self.input_name, int(self.input_device))
        self.context.set_tensor_address(self.output_name, int(self.output_device))

        # Execute inference
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        self.stream.synchronize()

        # Copy output back
        cuda.memcpy_dtoh_async(self.output_host, self.output_device, self.stream)
        self.stream.synchronize()

        # Convert output and post-process
        out_tensor = torch.from_numpy(self.output_host).cuda()
        out = out_tensor.view(self.output_shape)

        return out
