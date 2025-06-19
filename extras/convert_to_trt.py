### Ref - https://medium.com/@maxme006/how-to-create-a-tensorrt-engine-version-10-4-0-ec705013da7c
import os
import sys
import logging
import argparse
import numpy as np
import tensorrt as trt

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("EngineBuilder")


class EngineBuilder:
    """
    Parses an ONNX graph and builds a TensorRT engine from it.
    Optimized for maximum resource usage.
    """

    def __init__(self, verbose=False, workspace=16):
        """
        :param verbose: If enabled, a higher verbosity level will be set on the TensorRT logger.
        :param workspace: Max memory workspace to allow, in GB. Increased for maximum optimization.
        """
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            self.trt_logger.min_severity = trt.Logger.Severity.VERBOSE

        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")

        self.builder = trt.Builder(self.trt_logger)
        self.config = self.builder.create_builder_config()
        
        # Increase the workspace memory pool size for maximum performance
        self.config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, workspace * (2**30)
        )

        self.batch_size = None
        self.network = None
        self.parser = None

    def create_network(self, onnx_path):
        """
        Parse the ONNX graph and create the corresponding TensorRT network definition.
        :param onnx_path: The path to the ONNX graph to load.
        """
        self.network = self.builder.create_network(1)
        self.parser = trt.OnnxParser(self.network, self.trt_logger)

        onnx_path = os.path.realpath(onnx_path)
        with open(onnx_path, "rb") as f:
            if not self.parser.parse(f.read()):
                log.error("Failed to load ONNX file: {}".format(onnx_path))
                for error in range(self.parser.num_errors):
                    log.error(self.parser.get_error(error))
                sys.exit(1)

        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
        outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]

        # Save input tensor and name for use in optimization profile
        self.input_tensor = inputs[0]
        self.input_name = self.input_tensor.name

        log.info("Network Description")
        for input in inputs:
            self.batch_size = input.shape[0]
            log.info(
                "Input '{}' with shape {} and dtype {}".format(
                    input.name, input.shape, input.dtype
                )
            )
        for output in outputs:
            log.info(
                "Output '{}' with shape {} and dtype {}".format(
                    output.name, output.shape, output.dtype
                )
            )
        # assert self.batch_size > 0
        if self.batch_size in (None, -1):
            log.warning("Dynamic batch size detected. Optimization profile may be required.")
        else:
            assert self.batch_size > 0


    def create_engine(self, engine_path, precision="fp16", use_int8=False):
        engine_path = os.path.realpath(engine_path)
        os.makedirs(os.path.dirname(engine_path) or ".", exist_ok=True)
        log.info("Building {} Engine in {}".format(precision, engine_path))

        if precision == "fp16":
            if not self.builder.platform_has_fast_fp16:
                log.warning("FP16 not supported")
            self.config.set_flag(trt.BuilderFlag.FP16)

        if use_int8:
            if not self.builder.platform_has_fast_int8:
                log.warning("INT8 not supported")
            else:
                self.config.set_flag(trt.BuilderFlag.INT8)

        # No optimization profile needed for fixed shape ONNX

        engine_bytes = self.builder.build_serialized_network(self.network, self.config)
        if engine_bytes is None:
            log.error("Failed to create engine")
            sys.exit(1)

        with open(engine_path, "wb") as f:
            log.info("Serializing engine to file: {:}".format(engine_path))
            f.write(engine_bytes)


def main(args):
    builder = EngineBuilder(args.verbose, args.workspace)
    builder.create_network(args.onnx)
    builder.create_engine(args.engine, args.precision, use_int8=args.use_int8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx", help="The input ONNX model file to load", required=True, default="model.onnx")
    parser.add_argument("-e", "--engine", help="The output path for the TRT engine", required=True, default="model.trt")
    parser.add_argument(
        "-p", "--precision", default="fp16", choices=["fp32", "fp16"], help="The precision mode to build in, either fp32/fp16"
    )
    parser.add_argument(
        "--use_int8", action="store_true", help="Enable INT8 precision mode (only if supported by the hardware)"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable more verbose log output")
    parser.add_argument(
        "-w", "--workspace", default=16, type=int, help="The max memory workspace size to allow in GB (default 16 GB for optimization)"
    )
    args = parser.parse_args()
    main(args)  