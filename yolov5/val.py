# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 model accuracy on a custom dataset

Usage:
    $ python path/to/val.py --weights yolov5s.pt --data coco128.yaml --img 640

Usage - formats:
    $ python path/to/val.py --weights yolov5s.pt                 # PyTorch
                                      yolov5s.torchscript        # TorchScript
                                      yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                      yolov5s.xml                # OpenVINO
                                      yolov5s.engine             # TensorRT
                                      yolov5s.mlmodel            # CoreML (macOS-only)
                                      yolov5s_saved_model        # TensorFlow SavedModel
                                      yolov5s.pb                 # TensorFlow GraphDef
                                      yolov5s.tflite             # TensorFlow Lite
                                      yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.general import (LOGGER, check_dataset, check_img_size, check_requirements, check_yaml,
                           coco80_to_coco91_class, colorstr, increment_path, non_max_suppression, print_args,
                           scale_coords, xywh2xyxy, xyxy2xywh)
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, time_sync

from utils.calibrator import Yolov5EntropyCalibrator
import pycuda.driver as cuda
import tensorrt as trt
from utils.general import non_max_suppression_np
from utils import common



def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({
            'image_id': image_id,
            'category_id': class_map[int(p[5])],
            'bbox': [round(x, 3) for x in b],
            'score': round(p[4], 5)})


def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return correct


@torch.no_grad()
def run(
        data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        task='val',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        workers=8,  # max dataloader workers (per RANK in DDP mode)
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_img=False,
        save_conf=False,  # save confidences in --save-txt labels
        save_json=False,  # save a COCO-JSON results file
        project=ROOT / 'runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=False,  # use FP16 half-precision inference
        int8=False,
        dnn=False,  # use OpenCV DNN for ONNX inference
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=False,
        callbacks=Callbacks(),
        compute_loss=None,
        evalutate=False,
        calib_dataset_path='',
        calib_cache_path='',
        evaluate=False,
):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir


        yolo_model_onnx_path = '/home/sony/projects/weights/yolov5l/best.onnx'
        logger = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(logger)


        profile_shape = (1, 3, imgsz, imgsz)

        profile = builder.create_optimization_profile()
        profile.set_shape("images", profile_shape, profile_shape, profile_shape)

        config = builder.create_builder_config()
        config.add_optimization_profile(profile)
        config.max_workspace_size = common.GiB(1)
        flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        network = builder.create_network(flag)
        parser = trt.OnnxParser(network, logger)

        if not parser.parse_from_file(str(yolo_model_onnx_path)):
            raise RuntimeError(f'failed to load ONNX file: {yolo_model_onnx_path}')

        # fp16 mode
        if builder.platform_has_fast_fp16 and half:
            config.set_flag(trt.BuilderFlag.FP16)

        # int8 mode 
        elif builder.platform_has_fast_int8 and int8 :
            config.set_flag(trt.BuilderFlag.INT8)

            calib = Yolov5EntropyCalibrator(calib_dataset_path, calib_cache_path, profile_shape)
            config.int8_calibrator = calib

        engine = builder.build_engine(network, config)
        context = engine.create_execution_context()

        data = check_dataset(data)  # check


        nc = 1 if single_cls else int(data['nc'])  # number of classes
        iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
        niou = iouv.numel()
        names = data['names']

    # Dataloader
    if not training:
        # if pt and not single_cls:  # check --weights are trained on --data
        #     ncm = model.model.nc
        #     assert ncm == nc, f'{weights[0]} ({ncm} classes) trained on different --data than what you passed ({nc} ' \
        #                       f'classes). Pass correct combination of --weights and --data that are trained together.'
        # model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup
        # pad = 0.0 if task in ('speed', 'benchmark') else 0.5
        # rect = False if task == 'benchmark' else pt  # square inference for benchmarks
        # task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader = create_dataloader(data[task],
                                       imgsz,
                                       batch_size,
                                       32,
                                       single_cls,
                                       pad=0.5,
                                       rect=True,
                                       workers=workers,
                                       prefix=colorstr(f'{task}: '))[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    #names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}

    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    callbacks.run('on_val_start')
    pbar = tqdm(dataloader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
    for batch_i, (img, targets, paths, shapes) in enumerate(pbar):
        callbacks.run('on_val_batch_start')
        t1 = time_sync()
        
        img = img.cpu().numpy()
        img = img.astype(np.float32)
        targets = targets.to(device)
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = img.shape  # batch size, channels, height, width
        t2 = time_sync()
        dt[0] += t2 - t1

        stream = cuda.Stream()
        bindings = []

        #with torch.no_grad():

        t0 = time_sync()

        for binding in engine :

            if engine.binding_is_input(binding) : 
                context.set_binding_shape(0, profile_shape)
                shape = context.get_binding_shape(0)
                size = trt.volume(shape)
                dtype = trt.nptype(engine.get_binding_dtype(binding))

                host_mem = cuda.pagelocked_empty(size, dtype).reshape(shape)
                device_mem = cuda.mem_alloc(host_mem.nbytes)

                input = common.HostDeviceMem(host_mem, device_mem)

            elif binding == 'output' :
                size = trt.volume(engine.get_binding_shape(binding))
                dtype = trt.nptype(engine.get_binding_dtype(binding))

                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)

                output = common.HostDeviceMem(host_mem, device_mem)

            bindings.append(int(device_mem))

        np.copyto(input.host, img)
        # Run model
        t1 = time_sync()

        dt[0] += t1 - t0
        trt_outputs, dt_inf = common.do_inference_v2(context, bindings, input, output, stream)  # inference and training outputs
        t2 = time_sync()
        dt[1] += dt_inf
        t3 = time_sync()

        trt_outputs = trt_outputs.reshape((1, -1, nc + 5))
        outputs = non_max_suppression_np(trt_outputs, conf_thres, iou_thres, False)
        dt[2] += time_sync() - t3

        seen += 1


        if evaluate :         

            # Statistics per image
            for si, pred in enumerate(outputs):
                labels = targets[targets[:, 0] == si, 1:]
                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []  # target class
                path = Path(paths[si])
                #seen += 1

                if len(pred) == 0:
                    if nl:
                        stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                    continue

                # Predictions
                predn = pred.clone()
                scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

                # Append to text file
                if save_txt:
                    gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                    for *xyxy, conf, cls in predn.tolist():
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                # W&B logging - Media Panel Plots
                if len(wandb_images) < log_imgs and wandb_logger.current_epoch > 0:  # Check for test operation
                    if wandb_logger.current_epoch % wandb_logger.bbox_interval == 0:
                        box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                                    "class_id": int(cls),
                                    "box_caption": "%s %.3f" % (names[cls], conf),
                                    "scores": {"class_score": conf},
                                    "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
                        boxes = {"predictions": {"box_data": box_data, "class_labels": names}}  # inference-space
                        wandb_images.append(wandb_logger.wandb.Image(img[si], boxes=boxes, caption=path.name))
                wandb_logger.log_training_progress(predn, path, names) if wandb_logger and wandb_logger.wandb_run else None

                # Append to pycocotools JSON dictionary
                if save_json:
                    # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                    box = xyxy2xywh(predn[:, :4])  # xywh
                    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                    for p, b in zip(pred.tolist(), box.tolist()):
                        jdict.append({'image_id': image_id,
                                    'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
                                    'bbox': [round(x, 3) for x in b],
                                    'score': round(p[4], 5)})

                # Assign all predictions as incorrect
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
                if nl:
                    detected = []  # target indices
                    tcls_tensor = labels[:, 0]

                    # target boxes
                    tbox = xywh2xyxy(labels[:, 1:5])
                    scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                    if plots:
                        confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))

                    # Per target class
                    for cls in torch.unique(tcls_tensor):
                        ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                        pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                        # Search for detections
                        if pi.shape[0]:
                            # Prediction to target ious
                            ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                            # Append detections
                            detected_set = set()
                            for j in (ious > iouv[0]).nonzero(as_tuple=False):
                                d = ti[i[j]]  # detected target
                                if d.item() not in detected_set:
                                    detected_set.add(d.item())
                                    detected.append(d)
                                    correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                    if len(detected) == nl:  # all targets already located in image
                                        break

                # Append statistics (correct, conf, pcls, tcls)
                stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

            # Plot images
            if plots and batch_i < 3:
                f = save_dir / f'test_batch{batch_i}_labels.jpg'  # labels
                Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()
                f = save_dir / f'test_batch{batch_i}_pred.jpg'  # predictions
                Thread(target=plot_images, args=(img, output_to_target(outputs), paths, f, names), daemon=True).start()

            # Compute statistics
            stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
            if len(stats) and stats[0].any():
                p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
                ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
                mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
                nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
            else:
                nt = torch.zeros(1)

            # Print results
            pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
            print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print speeds
    if not training:
        print('Speed: inference {} ms NMS {} ms image {} ms per 1 image'.format(((dt[1] / seen) * 1E3), ((dt[2] / seen) * 1E3), ((dt[0] / seen) * 1E3)))


    # Print results per class
    # if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
    #     for i, c in enumerate(ap_class):
    #         print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))


    # # Plots
    # if plots:
    #     confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
    #     if wandb_logger and wandb_logger.wandb:
    #         val_batches = [wandb_logger.wandb.Image(str(f), caption=f.name) for f in sorted(save_dir.glob('test*.jpg'))]
    #         wandb_logger.log({"Validation": val_batches})
    # if wandb_images:
    #     wandb_logger.log({"Bounding Box Debugger/Images": wandb_images})

    # Save JSON
    # if save_json and len(jdict):
    #     w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
    #     anno_json = './coco/annotations/instances_val2017.json'  # annotations json
    #     pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
    #     print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
    #     with open(pred_json, 'w') as f:
    #         json.dump(jdict, f)

        # try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
        #     from pycocotools.coco import COCO
        #     from pycocotools.cocoeval import COCOeval

        #     anno = COCO(anno_json)  # init annotations api
        #     pred = anno.loadRes(pred_json)  # init predictions api
        #     eval = COCOeval(anno, pred, 'bbox')
        #     if is_coco:
        #         eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
        #     eval.evaluate()
        #     eval.accumulate()
        #     eval.summarize()
        #     map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        # except Exception as e:
        #     print(f'pycocotools unable to run: {e}')

    # Return results
    # model.float()  # for training
    # if not training:
    #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    #     print(f"Results saved to {save_dir}{s}")
    # maps = np.zeros(nc) + map
    # for i, c in enumerate(ap_class):
    #     maps[c] = ap[i]
    # return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t




def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/waymo.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5l.pt', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.45, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.25, help='NMS IoU threshold')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save_img', default=False)
    parser.add_argument('--save_txt', default=False)
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save_conf', default=False)
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT / '../runs/onnx_trt_detect', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--save_dir', default='/home/youngjin/projects/runs/onnx_trt_detect/')
    parser.add_argument('-d', '--calib_dataset_path', default='/home/youngjin/datasets/coco/train/images',
        help='path to the calibration dataset')
    parser.add_argument('-c', '--calib_cache_path', default='/home/youngjin/projects/runs/calib_cache/coco_calib_cache.cache',
        help='path to the calibration dataset')
    parser.add_argument('--half', default=False)
    parser.add_argument('--int8', default=False)
    parser.add_argument('--evaluate', default=False)




    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))

    if opt.task in ('train', 'val', 'test'):  # run normally
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.info(f'WARNING: confidence threshold {opt.conf_thres} >> 0.001 will produce invalid mAP values.')
        run(**vars(opt))

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = True  # FP16 for fastest results
        if opt.task == 'speed':  # speed benchmarks
            # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)

        elif opt.task == 'study':  # speed vs mAP benchmarks
            # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
            for opt.weights in weights:
                f = f'study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt'  # filename to save to
                x, y = list(range(256, 1536 + 128, 128)), []  # x axis (image sizes), y axis
                for opt.imgsz in x:  # img-size
                    LOGGER.info(f'\nRunning {f} --imgsz {opt.imgsz}...')
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)  # results and times
                np.savetxt(f, y, fmt='%10.4g')  # save
            os.system('zip -r study.zip study_*.txt')
            plot_val_study(x=x)  # plot


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
