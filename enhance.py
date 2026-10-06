import cv2
import numpy as np

def negative(img):
    return cv2.bitwise_not(img)

def log_transform(img):
    img_f = img.astype(np.float32)
    c = 255 / np.log(1 + np.max(img_f))
    log_image = c * (np.log(img_f + 1))
    return np.array(log_image, dtype=np.uint8)

def gamma_correct(img, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def equalize_color(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

def clahe_color(img, clip=2.5, tiles=8):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def smooth(img, k):
    return cv2.GaussianBlur(img, (k, k), 0)

def sharpen_laplacian(img):
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    return cv2.addWeighted(img, 1.5, laplacian, -0.5, 0)

def sobel_edges(gray):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    abs_sobelx = cv2.convertScaleAbs(sobelx)
    abs_sobely = cv2.convertScaleAbs(sobely)
    return cv2.addWeighted(abs_sobelx, 0.5, abs_sobely, 0.5, 0)

def canny_edges(gray):
    return cv2.Canny(gray, 100, 200)

def lbp_image(gray):
    # Vectorized 8-neighbour Local Binary Pattern
    out = np.zeros_like(gray, dtype=np.uint8)
    
    # Pad the image by 1 pixel to handle boundaries
    padded = np.pad(gray, ((1, 1), (1, 1)), mode='constant')
    
    # Compare center pixel with its 8 neighbors
    center = padded[1:-1, 1:-1]
    out |= ((padded[0:-2, 0:-2] >= center) << 7).astype(np.uint8) # Top-Left
    out |= ((padded[0:-2, 1:-1] >= center) << 6).astype(np.uint8) # Top
    out |= ((padded[0:-2, 2:] >= center) << 5).astype(np.uint8)   # Top-Right
    out |= ((padded[1:-1, 2:] >= center) << 4).astype(np.uint8)   # Right
    out |= ((padded[2:, 2:] >= center) << 3).astype(np.uint8)     # Bottom-Right
    out |= ((padded[2:, 1:-1] >= center) << 2).astype(np.uint8)   # Bottom
    out |= ((padded[2:, 0:-2] >= center) << 1).astype(np.uint8)   # Bottom-Left
    out |= ((padded[1:-1, 0:-2] >= center) << 0).astype(np.uint8) # Left
    
    return out

def mean_brightness(img):
    downscaled = cv2.resize(img, (64, 36))
    gray = cv2.cvtColor(downscaled, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

def low_light_fix(img, gamma=0.6):
    img_gamma = gamma_correct(img, gamma)
    return clahe_color(img_gamma)

def apply_mode(img, mode, dark_threshold=70):
    was_fixed = False
    if mode == "gamma":
        out = gamma_correct(img, 0.6)
        was_fixed = True
    elif mode == "clahe":
        out = clahe_color(img)
        was_fixed = True
    elif mode == "hist":
        out = equalize_color(img)
        was_fixed = True
    elif mode == "auto":
        brightness = mean_brightness(img)
        if brightness < dark_threshold:
            out = low_light_fix(img)
            was_fixed = True
        else:
            out = img.copy()
    else: # mode == "none"
        out = img.copy()
    
    return out, was_fixed

if __name__ == "__main__":
    # Smoke-test
    test_img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    
    assert negative(test_img).shape == (480, 640, 3)
    assert log_transform(test_img).shape == (480, 640, 3)
    assert gamma_correct(test_img, 0.5).shape == (480, 640, 3)
    assert equalize_color(test_img).shape == (480, 640, 3)
    assert clahe_color(test_img).shape == (480, 640, 3)
    assert smooth(test_img, 5).shape == (480, 640, 3)
    assert sharpen_laplacian(test_img).shape == (480, 640, 3)
    assert sobel_edges(test_gray).shape == (480, 640)
    assert canny_edges(test_gray).shape == (480, 640)
    assert lbp_image(test_gray).shape == (480, 640)
    b = mean_brightness(test_img)
    print(f"Mean brightness: {b}")
    assert low_light_fix(test_img).shape == (480, 640, 3)
    
    for mode in ["none", "gamma", "clahe", "hist", "auto"]:
        out, fixed = apply_mode(test_img, mode)
        assert out.shape == (480, 640, 3)
    
    print("enhance.py tests passed!")

