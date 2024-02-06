from rembg import remove
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
from collections import defaultdict
import os


def _make_stickers_folder(stickers_root_dir, classes_name):
    if not os.path.exists(stickers_root_dir):
        os.makedirs(stickers_root_dir)
        for c in classes_name:
            sub_folder_path = os.path.join(stickers_root_dir, c)
            os.makedirs(sub_folder_path)
            print(f"Subfolder '{sub_folder_path}' created successfully.")
    else:
        print(f"Folder '{stickers_root_dir}' already exists.")



def cut_bboxes(training_annotation_path, training_images_dir, original_stickers_root_dir):
    classes_name =  ["aeroplane", "bicycle", "bird", "boat", "bottle", 
                 "bus", "car", "cat", "chair", "cow", "diningtable", 
                 "dog", "horse", "motorbike", "person", "pottedplant", 
                 "sheep", "sofa", "train","tvmonitor"]
    
    # 使用defaultdict初始化，設置默認值為int的0
    class_counts = defaultdict(int)
    image_name_dict = defaultdict(None)
    
    # 建立stickers資料夾
    _make_stickers_folder(original_stickers_root_dir, classes_name)
    # rmbg_stickers_root_dir = './bboxs/rmbg_stickers/'
    # _make_stickers_folder(rmbg_stickers_root_dir, classes_name)


    # 讀取 pascal
    input_file = open(training_annotation_path, 'r')
    for line in input_file:
        line = line.strip()
        info = line.split(' ')

        # image name
        image_name = info[0]
        image_path = training_images_dir + image_name
        # print("image_path:", image_path, end='\r')
        # display.display(display.Image(image_path, width=200, height=200))
        image = cv.imread(image_path)
        
        # bounding boxs
        bounding_boxes = [int(num) for num in info[1:]]
        bounding_boxes = np.array(bounding_boxes).reshape((-1, 5))
        # bounding_boxes = [bounding_boxes[i:i + 5] for i in range(0, len(bounding_boxes), 5)]
        
        image_name_dict[image_name] = bounding_boxes.astype(float)
        
        for i, bbox in enumerate(bounding_boxes):
            x_min, y_min, x_max, y_max, class_label = bbox
            
            # 產生 stickers
            cropped_image = image[y_min:y_max, x_min:x_max]
            cv.imwrite(f'./{original_stickers_root_dir}/{classes_name[class_label]}/{image_name[:-4]}_{i+1}.jpg', cropped_image)
            
            # # 去除 stickers 背景
            # cropped_image_rmbg = remove(cropped_image)
            # cv.imwrite(f'./{rmbg_stickers_root_dir}/{classes_name[class_label]}/{image_name[:-4]}_{i+1}.jpg', cropped_image_rmbg)
            
            # print(classes_name[class_label])
            class_counts[classes_name[class_label]] += 1

    return class_counts, image_name_dict


def count_bboxes(training_annotation_path, training_images_dir):
    classes_name =  ["aeroplane", "bicycle", "bird", "boat", "bottle", 
                 "bus", "car", "cat", "chair", "cow", "diningtable", 
                 "dog", "horse", "motorbike", "person", "pottedplant", 
                 "sheep", "sofa", "train","tvmonitor"]
    
    # 使用defaultdict初始化，設置默認值為int的0
    class_counts = defaultdict(int)
    image_name_dict = defaultdict(None)
    

    # 讀取 pascal
    input_file = open(training_annotation_path, 'r')
    for line in input_file:
        line = line.strip()
        info = line.split(' ')

        # image name
        image_name = info[0]
        image_path = training_images_dir + image_name
        # print("image_path:", image_path, end='\r')
        # display.display(display.Image(image_path, width=200, height=200))
     
        # bounding boxs
        bounding_boxes = [int(num) for num in info[1:]]
        bounding_boxes = np.array(bounding_boxes).reshape((-1, 5))
        # bounding_boxes = [bounding_boxes[i:i + 5] for i in range(0, len(bounding_boxes), 5)]
        # print(bounding_boxes)
        image_name_dict[image_name] = bounding_boxes.astype(float)
        
        for i, bbox in enumerate(bounding_boxes):
            x_min, y_min, x_max, y_max, class_label = bbox
            # print(classes_name[class_label])
            class_counts[classes_name[class_label]] += 1

    return class_counts, image_name_dict


 
def _add_alpha_channel(img):
    """ Add an alpha channel to a jpg image."""
 
    b_channel, g_channel, r_channel = cv.split(img)  # Split channels of the jpg image
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255  # Create an Alpha channel
 
    img_new = cv.merge((b_channel, g_channel, r_channel, alpha_channel))  # Merge channels
    return img_new


def merge_images_with_transparency(background, foreground_with_alpha, top, bottom, left, right):
    """將具有透明通道的前景 PNG 圖像合併到背景 JPG 圖像上
       top, bottom, left, right 為叠加位置坐座標值
    """

    # 檢查背景圖像是否為 4 通道
    if background.shape[2] == 3:
        background = _add_alpha_channel(background)
    
    '''
    當進行圖像合併時，由於位置可能設置不當，可能會導致前景 PNG 圖像的邊界超過背景 JPG 圖像，
    而導致程序報錯。這裡設定一系列叠加位置的限制，以便在前景 PNG 圖像超出背景 JPG 圖像範圍時仍能正常進行合併。
    '''
    top_limit, bottom_limit, left_limit, right_limit = 0, foreground_with_alpha.shape[0], 0, foreground_with_alpha.shape[1]
 
    if left < 0:
        left_limit = -left
        left = 0
    if top < 0:
        top_limit = -top
        top = 0
    if right > background.shape[1]:
        right_limit = foreground_with_alpha.shape[1] - (right - background.shape[1])
        right = background.shape[1]
    if bottom > background.shape[0]:
        bottom_limit = foreground_with_alpha.shape[0] - (bottom - background.shape[0])
        bottom = background.shape[0]
 
    # 獲取要覆蓋圖像的透明度值，將像素值除以 255，使值保持在 0-1 之間
    alpha_foreground = foreground_with_alpha[top_limit:bottom_limit, left_limit:right_limit, 3] / 255.0
    alpha_background = 1 - alpha_foreground
    
    image_copy = background.copy()


    # 開始合併
    for color_channel in range(0, 3):
        image_copy[top:bottom, left:right, color_channel] = (
            (alpha_background * background[top:bottom, left:right, color_channel]) +
            (alpha_foreground * foreground_with_alpha[top_limit:bottom_limit, left_limit:right_limit, color_channel])
        )
 
    return image_copy



def copy_from_image(input_path, output_path): # input could be jpg file, but output should be png file
    """ 將 sticker 去背存檔"""
    
    input = cv.imread(input_path)
    output = remove(input)
    cv.imwrite(output_path, output)
    # plt.figure(figsize=(10,6))
    # plt.subplot(1,2,1)
    # plt.imshow(cv.cvtColor(input, cv.COLOR_BGR2RGB))
    # plt.subplot(1,2,2)
    # plt.imshow(cv.cvtColor(output, cv.COLOR_BGR2RGB))
    # plt.show()
    return output


def _resize_with_aspect_ratio(image, new_width):
    # 獲取原始圖片的高度、寬度和通道數
    height, width = image.shape[:2]
    
    # 計算新的高度，以保持原始寬高比
    aspect_ratio = width / height
    new_height = int(new_width / aspect_ratio)
    
    # 使用cv2.resize調整圖片大小
    resized_image = cv.resize(image, (new_width, new_height))
    
    return resized_image


def paste_image(background_path, sticker_path, new_img_path, new_width_sticker, x_pos, y_pos):
    """將去背的 sticker 貼到背景圖片"""
    
    background = cv.imread(background_path)
    sticker = cv.imread(sticker_path, cv.IMREAD_UNCHANGED)
    resized_sticker = _resize_with_aspect_ratio(sticker, new_width_sticker)
    sticker_x_min = x_pos
    sticker_x_max = x_pos + resized_sticker.shape[0]
    sticker_y_min = y_pos
    sticker_y_max = y_pos + resized_sticker.shape[1]
    new_img = merge_images_with_transparency(background, resized_sticker, sticker_x_min, sticker_x_max, sticker_y_min, sticker_y_max)
    cv.imwrite(new_img_path, new_img)
    # plt.figure(figsize=(10,6))
    # plt.subplot(1,3,1)
    # plt.imshow(cv.cvtColor(background, cv.COLOR_BGR2RGB))
    # plt.subplot(1,3,2)
    # plt.imshow(cv.cvtColor(sticker, cv.COLOR_BGR2RGB))
    # plt.subplot(1,3,3)
    # plt.imshow(cv.cvtColor(new_img, cv.COLOR_BGR2RGB))
    # plt.show()
    return new_img, sticker_x_min, sticker_x_max, sticker_y_min, sticker_y_max
    # return new_img, x_min,



def copy_and_paste(background_path, origin_sticker_path, sticker_path, output_path, new_width_sticker, x_pos, y_pos):
    copy_from_image(origin_sticker_path, sticker_path) # remove_background from input_path and save it into sticker path
    new_img, sticker_x_min, sticker_x_max, sticker_y_min, sticker_y_max = paste_image(background_path = background_path, sticker_path = sticker_path, new_img_path = output_path, new_width_sticker=300, x_pos = 155, y_pos = 200)
    return new_img, sticker_x_min, sticker_x_max, sticker_y_min, sticker_y_max


if __name__=="__main__":
    # background_path = './dataset/VOCdevkit_train/VOC2007/JPEGImages/000009.jpg'
    # origin_sticker_path = './dataset/VOCdevkit_train/VOC2007/JPEGImages/000007.jpg'
    # sticker_path = './output_car.png'
    # new_img_path = './new_img.jpg'
    # new_width_sticker = 300
    # x_pos = 155
    # y_pos = 200
    # # _ = _paste_image(background_path = background_path,sticker_path = sticker_path ,new_img_path = new_img_path , new_width_sticker=300, x_pos = 155, y_pos = 200)
    # new_img, sticker_x_min, sticker_x_max, sticker_y_min, sticker_y_max = copy_and_paste(background_path, origin_sticker_path, sticker_path, new_img_path, new_width_sticker, x_pos, y_pos)
    # plt.imshow(cv.cvtColor(new_img, cv.COLOR_BGR2RGB))
    # plt.show()
    
    
    training_annotation_path = './dataset/pascal_voc_training_data.txt'
    training_images_dir = './dataset/VOCdevkit_train/VOC2007/JPEGImages/'
    original_stickers_root_dir = './bboxs/original_stickers/'
    cut_bboxes(training_annotation_path, training_images_dir, original_stickers_root_dir)
    # count_bboxes(training_annotation_path, training_images_dir)

