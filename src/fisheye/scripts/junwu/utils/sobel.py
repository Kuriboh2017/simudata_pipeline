import cv2

def sobel_mask_cv(depth, thres):
    grad_X = cv2.Sobel(depth, ddepth=-1, dx=1, dy=0)
    grad_Y = cv2.Sobel(depth, ddepth=-1, dx=0, dy=1)
    gradient = (grad_X**2 + grad_Y**2)**0.5
    edge_mask = gradient > thres
    return edge_mask