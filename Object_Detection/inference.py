import time
import os
from pathlib import Path

import cv2
import torch
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, \
    scale_coords, xyxy2xywh, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized, TracedModel
import warnings


mdl_path = 'E:/Devanagari Number Plate Recognition/Project_All/Project_Host/Streamlit_App/Models/numberplate/best.pt'
img_path = 'E:/Devanagari Number Plate Recognition/Project_All/Project_Host/Streamlit_App/img/numberplate/plate-537.jpg'
extracted_path = 'E:/Devanagari Number Plate Recognition/Project_All/Project_Host/Streamlit_App/inference/extracted_numberplate/'


def detect(save_img=bool, trace=bool, save_conf=bool, save_txt=bool, view_img=bool, nosave=bool, augment=bool, agnostic_nms=bool,classes=list):
    source, weights, view_img, imgsz, trace, nosave, save_txt, augment, agnostic_nms, classes = img_path, mdl_path, view_img, 640, trace, nosave, save_txt, augment, agnostic_nms, classes
    save_img = not nosave and not source.endswith('.txt')  # save inference images

    # Directories
    save_dir = Path(extracted_path)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    output_dir = "E:/Devanagari Number Plate Recognition/Project_All/Project_Host/Streamlit_App/inference/numberplate/" # for the extracted boxes (added later on)
    box_idx = 0  # Initialize box index

    # Initialize
    set_logging()
    device = select_device('cpu')
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, imgsz)

    if half:
        model.half()  # to FP16

    # Set Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=augment)[0]
        t2 = time_synchronized()

        # Apply NMS

        conf_thres = 0.25
        iou_thres = 0.45
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        t3 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / "plate.jpg")  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    # ------------------------------------------------------------------------------------------ #
                    # Extract and save the detected box as a separate image using OpenCV
                    x1, y1, x2, y2 = [int(coord) for coord in xyxy]
                    box_image = im0[y1:y2, x1:x2]
                    box_image_path = os.path.join(output_dir, f"box_{box_idx}.jpg") 
                    cv2.imwrite(box_image_path, box_image)  # Save using OpenCV
                    # ------------------------------------------------------------------------------------------ #
                    # -------------------------- Extracting the digits ----------------------------------------- #
                    # ------------------------------------------------------------------------------------------ #

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=5)

                    
                    

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


def extract_numberplate():
    with torch.no_grad():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            detect(save_img=True, trace=True, save_conf=True, save_txt=False, view_img=False, nosave=False, augment=True, agnostic_nms=True, classes=[0])

extract_numberplate()