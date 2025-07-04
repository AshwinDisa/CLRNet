import time
import cv2
import torch
from tqdm import tqdm
import pytorch_warmup as warmup
import numpy as np
import random
import os
import glob
import copy

from clrnet.models.registry import build_net
from .registry import build_trainer, build_evaluator
from .optimizer import build_optimizer
from .scheduler import build_scheduler
from clrnet.datasets import build_dataloader
from clrnet.utils.recorder import build_recorder
from clrnet.utils.net_utils import save_model, load_network, resume_network
from mmcv.parallel import MMDataParallel

import pdb
import onnxruntime as ort
from clrnet.utils.visualization import imshow_lanes 
from clrnet.engine.tensorRT import TRTInference  
from clrnet.engine.ONNXRuntime import ONNXInference 

class Runner(object):
    def __init__(self, cfg):
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        self.cfg = cfg
        self.recorder = build_recorder(self.cfg)
        self.net = build_net(self.cfg)
        self.net = MMDataParallel(self.net,
                                  device_ids=range(self.cfg.gpus)).cuda()
        self.recorder.logger.info('Network: \n' + str(self.net))
        self.resume()
        self.optimizer = build_optimizer(self.cfg, self.net)
        self.scheduler = build_scheduler(self.cfg, self.optimizer)
        self.metric = 0.
        self.val_loader = None
        self.test_loader = None

    def to_cuda(self, batch):
        for k in batch:
            if not isinstance(batch[k], torch.Tensor):
                continue
            batch[k] = batch[k].cuda()
        return batch

    def resume(self):
        if not self.cfg.load_from and not self.cfg.finetune_from:
            return
        load_network(self.net, self.cfg.load_from, finetune_from=self.cfg.finetune_from, logger=self.recorder.logger)

    def train_epoch(self, epoch, train_loader):
        self.net.train()
        end = time.time()
        max_iter = len(train_loader)
        for i, data in enumerate(train_loader):
            if self.recorder.step >= self.cfg.total_iter:
                break
            date_time = time.time() - end
            self.recorder.step += 1
            data = self.to_cuda(data)
            output = self.net(data)
            self.optimizer.zero_grad()
            loss = output['loss'].sum()
            loss.backward()
            self.optimizer.step()
            if not self.cfg.lr_update_by_epoch:
                self.scheduler.step()
            batch_time = time.time() - end
            end = time.time()
            self.recorder.update_loss_stats(output['loss_stats'])
            self.recorder.batch_time.update(batch_time)
            self.recorder.data_time.update(date_time)

            if i % self.cfg.log_interval == 0 or i == max_iter - 1:
                lr = self.optimizer.param_groups[0]['lr']
                self.recorder.lr = lr
                self.recorder.record('train')

    def train(self):
        self.recorder.logger.info('Build train loader...')
        train_loader = build_dataloader(self.cfg.dataset.train,
                                        self.cfg,
                                        is_train=True)

        self.recorder.logger.info('Start training...')
        start_epoch = 0
        if self.cfg.resume_from:
            start_epoch = resume_network(self.cfg.resume_from, self.net,
                                         self.optimizer, self.scheduler,
                                         self.recorder)
        for epoch in range(start_epoch, self.cfg.epochs):
            self.recorder.epoch = epoch
            self.train_epoch(epoch, train_loader)
            if (epoch +
                    1) % self.cfg.save_ep == 0 or epoch == self.cfg.epochs - 1:
                self.save_ckpt()
            if (epoch +
                    1) % self.cfg.eval_ep == 0 or epoch == self.cfg.epochs - 1:
                self.validate()
            if self.recorder.step >= self.cfg.total_iter:
                break
            if self.cfg.lr_update_by_epoch:
                self.scheduler.step()


    def test(self):
        if not self.test_loader:
            self.test_loader = build_dataloader(self.cfg.dataset.test,
                                                self.cfg,
                                                is_train=False)
        self.net.eval()
        predictions = []
        for i, data in enumerate(tqdm(self.test_loader, desc=f'Testing')):
            data = self.to_cuda(data)

            with torch.no_grad():
                output = self.net(data)
                output = self.net.module.heads.get_lanes(output)
                predictions.extend(output)
            if self.cfg.view:
                self.test_loader.dataset.view(output, data['meta'])

            pdb.set_trace()

        metric = self.test_loader.dataset.evaluate(predictions,
                                                   self.cfg.work_dir)
        if metric is not None:
            self.recorder.logger.info('metric: ' + str(metric))

    def validate(self):
        if not self.val_loader:
            self.val_loader = build_dataloader(self.cfg.dataset.val,
                                               self.cfg,
                                               is_train=False)
        self.net.eval()
        predictions = []
        for i, data in enumerate(tqdm(self.val_loader, desc=f'Validate')):
            data = self.to_cuda(data)
            
            with torch.no_grad():
                output = self.net(data)
                output = self.net.module.heads.get_lanes(output)
                predictions.extend(output)
            if self.cfg.view:
                self.val_loader.dataset.view(output, data['meta'])

        metric = self.val_loader.dataset.evaluate(predictions,
                                                  self.cfg.work_dir)
        self.recorder.logger.info('metric: ' + str(metric))

    def scale_lane_points(self, coords_in, scale_x, scale_y):
        coords_in[:, 0] *= scale_x  # x
        coords_in[:, 1] *= scale_y  # y
        return coords_in

    def test_image_onnx(self, onnx_path):

        # img_path = '/home/ashd/projects/CLRNet/data/driver_100_30frame/05250358_0283.MP4/00000.jpg'
        # img_path = '/home/ashd/projects/CLRNet/data/driver_37_30frame/05181743_0267.MP4/00000.jpg'
        # img_path = '/home/ashd/projects/CLRNet/data/driver_37_30frame/05191535_0475.MP4/00005.jpg'
        # img_path = '/home/ashd/projects/CLRNet/data/driver_193_90frame/06042010_0511.MP4/00000.jpg'
        # img_path = '/home/ashd/projects/CLRNet/data/driver_100_30frame/05251517_0433.MP4/00690.jpg'
        # img_path = 'extras/test_images/frame_950.png'
        # img_path = 'extras/test_images/frame_2000.png'
        # img_path = 'extras/test_images/00510.jpg'
        img_path = 'extras/test_images/P3scene1/047.jpg'
        # img_path = 'extras/test_images/P3scene1/100.jpg'

        og_image = cv2.imread(img_path) # (980, 1280, 3) for P3

        image = cv2.resize(og_image, (1640, 590)) # (590, 1640, 3)

        # Step 1: CUT the top 270 pixels
        cut_height = 270
        cut_image = image[cut_height:, :, :]  # (320, 1640, 3)

        # Step 2: Resize to (800, 320)
        img_resize = cv2.resize(cut_image, (800, 320)) # (320, 800, 3)

        # Step 3: Convert to float32
        img = img_resize.astype(np.float32) / 255.0

        # Step 5: Prepare for ONNX - HWC to CHW then add batch dim
        img = np.transpose(img, (2, 0, 1))  # CHW
        img = np.expand_dims(img, axis=0).astype(np.float32)  # NCHW

        # Step 6: Run ONNX inference
        session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: img})
        output_tensor = torch.from_numpy(output[0]).cuda()

        # Step 7: Postprocess and visualize
        output = self.net.module.heads.get_lanes(output_tensor)

        lanes = []
        for lane in output[0]:
            lane_new = lane.to_array(self.cfg)
            coords_new = lane.scale_lane_points(copy.deepcopy(lane_new), from_size=(1640, 590), to_size=og_image.shape[:2][::-1])
            lanes.append(coords_new)

        # Optional: resize back image for visualization
        imshow_lanes(og_image, lanes, show=True)

    def infer(self, mode, model_path, image_dir=None):

        img_path = sorted(glob.glob(f"{image_dir}*.jpg"))
        cut_height = 270

        if mode == 'pytorch':
            # Pytorch
            checkpoint = torch.load(model_path, map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
            self.net.load_state_dict(checkpoint['net'], strict=False)
            self.net.eval()
            self.net.cuda()

        elif mode == 'onnx_python_cpu':
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            input_name = session.get_inputs()[0].name

        elif mode == 'onnx_python_gpu':
            inference_engine = ONNXInference(model_path)

        elif mode == 'tensorrt':
            trt_infer = TRTInference(model_path, self.net.module.heads.get_lanes)

        init_time = time.time()
        frames = 0 

        # set scale
        img = cv2.imread(img_path[0])
        scale_x = img.shape[1] / 1640
        scale_y = img.shape[0] / 590

        # image_folder = image_dir.split('/')[-2]
        # out_path = f'extras/test_images/infer_results/{image_folder}'

        for img in img_path:

            init_time_per_frame = time.time()

            og_image = cv2.imread(img)

            # trained on this config
            test_image = cv2.resize(og_image[int(cut_height * og_image.shape[0] / 590):], (800, 320))
            
            # float32 and normalize
            img = test_image.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]

            if mode == 'pytorch':
                output, infer_time = self.infer_pytorch(img)

            elif mode == 'onnx_python_cpu':
                output, infer_time = self.infer_onnx_python_cpu(img, session, input_name)
                
            elif mode == 'onnx_python_gpu':
                output, infer_time = self.infer_onnx_python_gpu(img, inference_engine)
            
            elif mode == 'tensorrt':
                output, infer_time = self.infer_tensorrt(img, trt_infer)
            
            else:
                raise ValueError("Mode must be 'pytorch', 'onnx_python_cpu', or 'onnx_python_gpu'.")

            lanes = []
            for lane in output[0]:
                lane_new = lane.to_array(self.cfg)
                coords_new = self.scale_lane_points(lane_new, scale_x, scale_y)
                lanes.append(coords_new)    

            final_time_per_frame = time.time()
            fps = int(1 / (final_time_per_frame - init_time_per_frame + 1e-8))
            
            # Show frame with FPS and time to infer
            imshow_lanes(og_image, lanes, show=True, video=True, fps=fps, infer_time=infer_time, mode=mode)
            
            # to save the image
            # imshow_lanes(og_image, lanes, show=True, out_file=out_path, video=True, fps=fps, infer_time=infer_time, mode=mode, frames=frames)
        
            frames += 1
            if time.time() - init_time > 20:
                print(f"Stopping after 20 seconds of inference for {mode}.")
                break

        print(f"Mean FPS with {mode}: ", 1/(time.time() - init_time + 1e-8) * frames)

    def infer_pytorch(self, img):

        img = torch.from_numpy(img).cuda()

        t0 = time.time()

        with torch.no_grad():
            output = self.net(img)
            output = self.net.module.heads.get_lanes(output)

        return output, time.time() - t0

    def infer_onnx_python_cpu(self, img, session, input_name):

        t0 = time.time()

        output = session.run(None, {input_name: img})
        output_tensor = torch.from_numpy(output[0]).cuda()
        output = self.net.module.heads.get_lanes(output_tensor)

        return output, time.time() - t0

    def infer_onnx_python_gpu(self, img, inference_engine):
            
        t0 = time.time()     
        output = inference_engine.infer(img)
        output_tensor = torch.from_numpy(output[0]).cuda()
        output = self.net.module.heads.get_lanes(output_tensor)

        return output, time.time() - t0
    
    def infer_tensorrt(self, img, trt_infer):

        t0 = time.time()
        output = trt_infer.infer(img)
        output = self.net.module.heads.get_lanes(output)

        return output, time.time() - t0

    def save_ckpt(self, is_best=False):
        save_model(self.net, self.optimizer, self.scheduler, self.recorder,
                   is_best)
