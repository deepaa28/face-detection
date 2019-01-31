import math
import numpy as np
import torch
from mtcnn.nets import PNet, RNet, ONet
from mtcnn.box_utils import nms, calibrate_box, get_image_boxes, convert_to_square, _preprocess
from PIL import Image
from PIL import ImageDraw

PATH_TRAINED_DATA_PNET = './data/pnet-trained-data.pth'
PATH_TRAINED_DATA_RNET = './data/rnet-trained-data.pth'
PATH_TRAINED_DATA_ONET = './data/onet-trained-data.pth'


def save_trained_weights(pnet, rnet, onet):
    torch.save(pnet.state_dict(), PATH_TRAINED_DATA_PNET)
    torch.save(rnet.state_dict(), PATH_TRAINED_DATA_RNET)
    torch.save(onet.state_dict(), PATH_TRAINED_DATA_ONET)


def load_trained_weights():
    pnet = PNet()
    pnet.load_state_dict(torch.load(PATH_TRAINED_DATA_PNET))
    pnet.eval()

    rnet = RNet()
    rnet.load_state_dict(torch.load(PATH_TRAINED_DATA_RNET))
    rnet.eval()

    onet = ONet()
    onet.load_state_dict(torch.load(PATH_TRAINED_DATA_ONET))
    onet.eval()

    return pnet, rnet, onet


"""
Since we jointly perform face detection and alignment, here
we use four different kinds of data annotation in our training process:
(i) Negatives: Regions that the Intersection-over-Union (IoU) ratio
less than 0.3 to any ground-truth faces
(ii) Positives: IoU above 0.65 to a ground truth face
(iii) Part faces: IoU between 0.4 and 0.65 to a ground truth face
(iv) Landmark faces: faces labeled 5 landmarks’ positions.
Negatives and positives are used for face classification tasks,
positives and part faces are used for bounding box regression,
and landmark faces are used for facial landmark localization. 
"""


def detect_faces(image,
                 min_face_size=20.0,
                 thresholds=[0.6, 0.7, 0.8],
                 nms_thresholds=[0.7, 0.7, 0.7]):
    pnet, rnet, onet = load_trained_weights()

    width, height = image.size
    min_length = min(height, width)
    min_detection_size = 12
    factor = math.sqrt(0.5)

    scales = []
    m = min_detection_size / min_face_size
    min_length *= m

    factor_count = 0
    while min_length > min_detection_size:
        scales.append(m * factor ** factor_count)
        min_length *= factor
        factor_count += 1

    # STAGE 1
    bounding_boxes = []
    for s in scales:  # run P-Net on different scales
        boxes = run_first_stage(image, pnet, scale=s, threshold=thresholds[0])
        bounding_boxes.append(boxes)
    bounding_boxes = [i for i in bounding_boxes if i is not None]
    bounding_boxes = np.vstack(bounding_boxes)

    keep = nms(bounding_boxes[:, 0:5], nms_thresholds[0])
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
    bounding_boxes = convert_to_square(bounding_boxes)
    bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

    # STAGE 2
    img_boxes = get_image_boxes(bounding_boxes, image, size=24)
    img_boxes = torch.FloatTensor(img_boxes)
    output = rnet(img_boxes)
    offsets = output[0].data.numpy()  # shape [n_boxes, 4]
    probs = output[1].data.numpy()  # shape [n_boxes, 2]

    keep = np.where(probs[:, 1] > thresholds[1])[0]
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
    offsets = offsets[keep]

    keep = nms(bounding_boxes, nms_thresholds[1])
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes = calibrate_box(bounding_boxes, offsets[keep])
    bounding_boxes = convert_to_square(bounding_boxes)
    bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

    # STAGE 3
    img_boxes = get_image_boxes(bounding_boxes, image, size=48)
    if len(img_boxes) == 0:
        return [], []
    img_boxes = torch.FloatTensor(img_boxes)
    output = onet(img_boxes)
    landmarks = output[0].data.numpy()  # shape [n_boxes, 10]
    offsets = output[1].data.numpy()  # shape [n_boxes, 4]
    probs = output[2].data.numpy()  # shape [n_boxes, 2]

    keep = np.where(probs[:, 1] > thresholds[2])[0]
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
    offsets = offsets[keep]
    landmarks = landmarks[keep]

    # compute landmark points
    width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
    height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
    xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
    landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(width, 1) * landmarks[:, 0:5]
    landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(height, 1) * landmarks[:, 5:10]

    bounding_boxes = calibrate_box(bounding_boxes, offsets)
    keep = nms(bounding_boxes, nms_thresholds[2], mode='min')
    bounding_boxes = bounding_boxes[keep]
    landmarks = landmarks[keep]

    return bounding_boxes, landmarks


def run_first_stage(image, net, scale, threshold):
    """ 
    Run P-Net, generate bounding boxes, and do NMS.
    """
    width, height = image.size
    sw, sh = math.ceil(width * scale), math.ceil(height * scale)
    img = image.resize((sw, sh), Image.BILINEAR)
    img = np.asarray(img, 'float32')
    img = torch.FloatTensor(_preprocess(img))

    output = net(img)
    probs = output[1].data.numpy()[0, 1, :, :]
    offsets = output[0].data.numpy()

    boxes = _generate_bboxes(probs, offsets, scale, threshold)
    if len(boxes) == 0:
        return None

    keep = nms(boxes[:, 0:5], overlap_threshold=0.5)
    return boxes[keep]


def _generate_bboxes(probs, offsets, scale, threshold):
    """
       Generate bounding boxes at places where there is probably a face.
    """
    stride = 2
    cell_size = 12

    inds = np.where(probs > threshold)

    if inds[0].size == 0:
        return np.array([])

    tx1, ty1, tx2, ty2 = [offsets[0, i, inds[0], inds[1]] for i in range(4)]

    offsets = np.array([tx1, ty1, tx2, ty2])
    score = probs[inds[0], inds[1]]

    # P-Net is applied to scaled images, so we need to rescale bounding boxes back
    bounding_boxes = np.vstack([
        np.round((stride * inds[1] + 1.0) / scale),
        np.round((stride * inds[0] + 1.0) / scale),
        np.round((stride * inds[1] + 1.0 + cell_size) / scale),
        np.round((stride * inds[0] + 1.0 + cell_size) / scale),
        score, offsets
    ])

    return bounding_boxes.T


def main():
    image = Image.open('test/images/delta2.jpeg')
    bounding_boxes, landmarks = detect_faces(image)
    image = show_bboxes(image, bounding_boxes, landmarks)
    image.show()


def show_bboxes(img, bounding_boxes, facial_landmarks=[]):
    """
        Draw bounding boxes and facial landmarks.
    """
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)

    for b in bounding_boxes:
        draw.rectangle([(b[0], b[1]), (b[2], b[3])],
                       outline='red')

    for p in facial_landmarks:
        for i in range(5):
            draw.ellipse([(p[i] - 1.0, p[i + 5] - 1.0),
                          (p[i] + 1.0, p[i + 5] + 1.0)],
                         outline='blue')
    return img_copy


if __name__ == "__main__":
    main()
