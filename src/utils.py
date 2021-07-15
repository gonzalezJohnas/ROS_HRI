



def check_img_limit(x1,y1,x2,y2, img_shape):
    y2 = 0 if y2 < 0 else y2
    y2 = img_shape[0] if y2 > img_shape[0] else y2

    x1 = 0 if x1 < 0 else x1
    x2 = img_shape[1] if x2 > img_shape[1] else x2

    return x1, y1, x2, y2