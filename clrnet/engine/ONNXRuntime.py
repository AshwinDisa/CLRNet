import onnxruntime as ort
import numpy as np
import time

class ONNXInference:
    def __init__(self, model_path):
        # More aggressive GPU optimization settings
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                'cudnn_conv_algo_search': 'EXHAUSTIVE',    # Find best conv algorithms
                'do_copy_in_default_stream': True,
                'cudnn_conv_use_max_workspace': True,
            })
        ]
        
        # Session options for performance
        sess_options = ort.SessionOptions()
        sess_options.enable_mem_pattern = False
        sess_options.enable_mem_reuse = False
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(model_path, providers=providers, sess_options=sess_options)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
    
    def infer(self, img):
        t0 = time.time()
        
        # Method 1: Using IO binding with proper arguments
        io_binding = self.session.io_binding()
        
        # Create OrtValue from numpy array
        input_ortvalue = ort.OrtValue.ortvalue_from_numpy(img, 'cuda', 0)
        
        # Bind input with the OrtValue directly
        io_binding.bind_ortvalue_input(self.input_name, input_ortvalue)
        
        # Bind outputs to GPU
        for output_name in self.output_names:
            io_binding.bind_output(output_name, 'cuda')
        
        # Run inference
        self.session.run_with_iobinding(io_binding)
        
        # Get outputs
        output = io_binding.get_outputs()
        output_numpy = output[0].numpy()

        return output_numpy, time.time() - t0
    
    def infer_alternative(self, img):
        """Alternative method without IO binding - simpler but might be slightly slower"""
        t0 = time.time()
        
        # Direct inference - ONNX Runtime will handle GPU transfers
        outputs = self.session.run(None, {self.input_name: img})
        
        return outputs[0], time.time() - t0