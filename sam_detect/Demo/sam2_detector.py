from sam_detect.Module.sam2_detector import SAM2Detector


def demo():
    model_file_path = "/home/chli/chLi/Model/SAM2/sam2.1_hiera_large.pt"
    device = "cuda:0"

    image_file_path = "/home/chli/下载/test_room_pic.jpeg"

    sam2_detector = SAM2Detector(model_file_path, device)
    sam2_detector.detectImageFile(image_file_path)
    return True
