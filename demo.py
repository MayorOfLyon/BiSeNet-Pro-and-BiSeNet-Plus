import cv2
import argparse
from model.build_BiSeNet import BiSeNet
import os
import torch
import cv2
from imgaug import augmenters as iaa
from PIL import Image
from torchvision import transforms
import numpy as np
from utils import reverse_one_hot, get_label_info, colour_code_segmentation
import time

def load_partial_model(model, checkpoint_path):
    # 加载模型的权重
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    # 获取模型当前的权重字典
    model_dict = model.state_dict()
    
    # 过滤掉不匹配的权重
    checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict and v.size() == model_dict[k].size()}
    
    # 更新模型权重字典
    model_dict.update(checkpoint)
    
    # 加载更新后的权重字典
    model.load_state_dict(model_dict)


def predict_on_image(model, args):
    # pre-processing on image
    image = cv2.imread(args.data, -1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resize = iaa.Scale({'height': args.crop_height, 'width': args.crop_width})
    resize_det = resize.to_deterministic()
    image = resize_det.augment_image(image)
    image = Image.fromarray(image).convert('RGB')
    image = transforms.ToTensor()(image)
    image = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image).unsqueeze(0)
    # read csv label path
    label_info = get_label_info(args.csv_path)
    # predict
    model.eval()
    predict = model(image).squeeze()
    predict = reverse_one_hot(predict)
    predict = colour_code_segmentation(np.array(predict.cpu()), label_info)
    predict = cv2.resize(np.uint8(predict), (960, 720))
    cv2.imwrite(args.save_path, cv2.cvtColor(np.uint8(predict), cv2.COLOR_RGB2BGR))

def predict_on_video(model, args):
    start_time = time.time()  # 记录函数开始时间

    cap = cv2.VideoCapture(args.data)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    label_info = get_label_info(args.csv_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(args.save_path, cv2.VideoWriter_fourcc(*'XVID'), 20, (frame_width, frame_height))
    number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        number += 1
        # Pre-process frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resize = iaa.Scale({'height': args.crop_height, 'width': args.crop_width})
        resize_det = resize.to_deterministic()
        frame = resize_det.augment_image(frame)
        frame = Image.fromarray(frame).convert('RGB')
        frame = transforms.ToTensor()(frame)
        frame = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(frame).unsqueeze(0)

        # Predict
        model.eval()
        predict = model(frame).squeeze()
        predict = reverse_one_hot(predict)
        predict = colour_code_segmentation(np.array(predict.cpu()), label_info)
        predict = cv2.resize(np.uint8(predict), (frame_width, frame_height))
        out.write(cv2.cvtColor(np.uint8(predict), cv2.COLOR_RGB2BGR))



    end_time = time.time()  # 记录函数结束时间
    elapsed_time = end_time - start_time  # 计算函数运行时间
    # 计算fps
    print("Total frames: ", number)
    print("Total time: {:.2f} seconds".format(elapsed_time))
    print("FPS: {:.2f}".format(number / elapsed_time))

def predict_on_camera(model, args):
    start_time = time.time()  # 记录函数开始时间

    cap = cv2.VideoCapture(0)  # 0表示第一个相机，1表示第二个相机，依此类推
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    label_info = get_label_info(args.csv_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(args.save_path, cv2.VideoWriter_fourcc(*'XVID'), 20, (frame_width, frame_height))
    number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        number += 1

        frame_for_disp = frame.copy()  # Copy the frame for display 
        # Pre-process frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
       

        resize = iaa.Scale({'height': args.crop_height, 'width': args.crop_width})
        resize_det = resize.to_deterministic()
        frame = resize_det.augment_image(frame)
        frame = Image.fromarray(frame).convert('RGB')
        frame = transforms.ToTensor()(frame)
        frame = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(frame).unsqueeze(0)

        # Predict
        model.eval()
        predict = model(frame).squeeze()
        predict = reverse_one_hot(predict)
        predict = colour_code_segmentation(np.array(predict.cpu()), label_info)
        predict = cv2.resize(np.uint8(predict), (frame_width, frame_height))
        out.write(cv2.cvtColor(np.uint8(predict), cv2.COLOR_RGB2BGR))

        # Combine original frame and prediction result
        combined_frame = np.concatenate((frame_for_disp, np.uint8(predict)), axis=1)

        # Display combined frame
        cv2.imshow('Camera and Prediction', combined_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


    end_time = time.time()  # 记录函数结束时间
    elapsed_time = end_time - start_time  # 计算函数运行时间
    # 计算fps
    print("Total frames: ", number)
    print("Total time: {:.2f} seconds".format(elapsed_time))
    print("FPS: {:.2f}".format(number / elapsed_time))



def main(params):
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', action='store_true', default=False, help='predict on image')
    parser.add_argument('--video', action='store_true', default=False, help='predict on video')
    parser.add_argument('--camera', action='store_true', default=False, help='predict on camera')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='The path to the pretrained weights of model')
    parser.add_argument('--context_path', type=str, default="resnet101", help='The context path model you are using.')
    parser.add_argument('--num_classes', type=int, default=12, help='num of object classes (with void)')
    parser.add_argument('--data', type=str, default=None, help='Path to image or video for prediction')
    parser.add_argument('--crop_height', type=int, default=720, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=960, help='Width of cropped/resized input image to network')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='Whether to user gpu for training')
    parser.add_argument('--csv_path', type=str, default=None, required=True, help='Path to label info csv file')
    parser.add_argument('--save_path', type=str, default=None, required=True, help='Path to save predict image')
    parser.add_argument('--model', type=str, default='BiSeNet-Pro', help='BiSeNet-Pro or BiSeNet-plus')

    args = parser.parse_args(params)

    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    model = BiSeNet(args.num_classes, args.context_path, args.model)
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    # load pretrained model if exists
    print('load model from %s ...' % args.checkpoint_path)
    # model.module.load_state_dict(torch.load(args.checkpoint_path))
    # model.load_state_dict(torch.load(args.checkpoint_path))
    load_partial_model(model.module if torch.cuda.is_available() and args.use_gpu else model, args.checkpoint_path)
    print('Done!')



    # predict on image
    if args.image:
        predict_on_image(model, args)

    # predict on video
    if args.video:
        # pass
        predict_on_video(model, args)

    # predict on camera
    if args.camera:
        predict_on_camera(model, args)



if __name__ == '__main__':
    # params = [
    #     '--image',
    #     '--data', './test.png',
    #     '--checkpoint_path', './checkpoints_18_sgd_camvid/best_dice_loss_version1.pth',
    #     '--cuda', '0',
    #     '--csv_path', './CamVid/class_dict.csv',
    #     '--save_path', '1.png',
    #     '--context_path', 'resnet18'
    # ]
    # main(params)

    # params = [
    #     '--video',
    #     '--data', './video.mp4',  # Path to the input video
    #     '--checkpoint_path', './checkpoints_18_sgd_camvid/best_dice_loss_version1.pth',
    #     '--cuda', '0',
    #     '--csv_path', './CamVid/class_dict.csv',
    #     '--save_path', '1.mp4',  # Path to save the output video
    #     '--context_path', 'resnet18'
    # ]
    # main(params)

    params = [
        '--camera',
        '--checkpoint_path', './checkpoints_18_sgd_camvid/best_dice_loss_1.pth',
        '--cuda', '0',
        '--csv_path', './CamVid/class_dict.csv',
        '--save_path', 'camera_output.mp4',  # Path to save the output video
        '--context_path', 'resnet18'
    ]
    main(params)


