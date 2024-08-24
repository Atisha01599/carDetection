import matplotlib.pyplot as plt
import numpy as np
import json
import cv2
import os


class SVM:
    def __init__(self, iterations=2000, lr=0.001, lambdaa=0.01):
        self.lambdaa = lambdaa
        self.iterations = iterations
        self.lr = lr
        self.w = None
        self.b = None

    def initialize_parameters(self, X):
        _, n = X.shape
        self.w = np.zeros(n)
        self.b = 0

    def gradient_descent(self, X, y):
        y_ = np.where(y <= 0, -1, 1)
        for i, x in enumerate(X):
            if y_[i] * (np.dot(x, self.w) - self.b) >= 1:
                dw = 2 * self.lambdaa * self.w
                db = 0
            else:
                dw = 2 * self.lambdaa * self.w - np.dot(x, y_[i])
                db = y_[i]
            self.update_parameters(dw, db)

    def update_parameters(self, dw, db):
        self.w = self.w - self.lr * dw
        self.b = self.b - self.lr * db

    def fit(self, X, y):
        self.initialize_parameters(X)
        for i in range(self.iterations):
            self.gradient_descent(X, y)

    def predict(self, X):
        output = np.dot(X, self.w) - self.b
        label_signs = np.sign(output)
        predictions = np.where(label_signs <= -1, 0, 1)
        return predictions

def accuracy_score(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = len(y_true)
    accuracy = correct_predictions / total_predictions
    return accuracy


def classification_report(y_true, y_pred):
    classes = np.unique(y_true)
    report = {}
    
    for cls in classes:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        tn = np.sum((y_true != cls) & (y_pred != cls))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        support = np.sum(y_true == cls)
        
        report[cls] = {
            'precision': precision,
            'recall': recall,
            'f1-score': f1_score,
            'support': support
        }

    # Formatting the report as a string
    report_str = "              precision    recall  f1-score   support\n\n"
    for cls in report:
        report_str += f"{cls:>12} {report[cls]['precision']:>10.2f} {report[cls]['recall']:>10.2f} {report[cls]['f1-score']:>10.2f} {report[cls]['support']:>10}\n"
    
    # Calculate averages
    avg_precision = np.mean([report[cls]['precision'] for cls in report])
    avg_recall = np.mean([report[cls]['recall'] for cls in report])
    avg_f1_score = np.mean([report[cls]['f1-score'] for cls in report])
    total_support = np.sum([report[cls]['support'] for cls in report])

    report_str += "\n    accuracy                           {:.2f}\n".format(np.sum(y_true == y_pred) / len(y_true))
    report_str += "   macro avg       {:.2f}      {:.2f}      {:.2f}      {:>10}\n".format(avg_precision, avg_recall, avg_f1_score, total_support)
    report_str += "weighted avg       {:.2f}      {:.2f}      {:.2f}      {:>10}\n".format(
        np.average([report[cls]['precision'] for cls in report], weights=[report[cls]['support'] for cls in report]),
        np.average([report[cls]['recall'] for cls in report], weights=[report[cls]['support'] for cls in report]),
        np.average([report[cls]['f1-score'] for cls in report], weights=[report[cls]['support'] for cls in report]),
        total_support
    )
    
    return report_str


def compute_gradients(image):
    gx = np.zeros(image.shape, dtype=np.float32)
    gy = np.zeros(image.shape, dtype=np.float32)

    gx[:, :-1] = np.diff(image, n=1, axis=1)
    gy[:-1, :] = np.diff(image, n=1, axis=0)

    magnitude = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx) * (180 / np.pi) % 180

    return magnitude, angle

def bilinear_interpolate(image, x, y):
    x0 = int(x)
    x1 = min(x0 + 1, image.shape[1] - 1)
    y0 = int(y)
    y1 = min(y0 + 1, image.shape[0] - 1)
    
    Ia = image[y0, x0]
    Ib = image[y1, x0]
    Ic = image[y0, x1]
    Id = image[y1, x1]
    
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)
    
    return wa * Ia + wb * Ib + wc * Ic + wd * Id

def resize_image(image, new_width, new_height):
    src_height, src_width = image.shape[:2]
    dst_image = np.zeros((new_height, new_width), dtype=image.dtype)
    
    x_ratio = src_width / new_width
    y_ratio = src_height / new_height
    
    for i in range(new_height):
        for j in range(new_width):
            x = j * x_ratio
            y = i * y_ratio
            dst_image[i, j] = bilinear_interpolate(image, x, y)
    
    return dst_image

def extract_hog_features(image, bbox):
    x, y, w, h = map(int, bbox)
    roi = image[y:y+h, x:x+w]
    # roi_resized = resize_image(roi, (64, 128))   # Resize using custom function
    roi_resized = resize_image(roi, 64, 128)
    # Parameters for HOG
    cell_size = (8, 8)
    block_size = (2, 2)
    nbins = 9

    # Compute gradients
    magnitude, angle = compute_gradients(roi_resized)

    # Create histogram of gradients
    cell_x, cell_y = cell_size
    num_cells_x = roi_resized.shape[1] // cell_x
    num_cells_y = roi_resized.shape[0] // cell_y
    histograms = np.zeros((num_cells_y, num_cells_x, nbins))

    for i in range(num_cells_y):
        for j in range(num_cells_x):
            cell_mag = magnitude[i*cell_y:(i+1)*cell_y, j*cell_x:(j+1)*cell_x]
            cell_ang = angle[i*cell_y:(i+1)*cell_y, j*cell_x:(j+1)*cell_x]
            hist, _ = np.histogram(cell_ang, bins=nbins, range=(0, 180), weights=cell_mag)
            histograms[i, j, :] = hist

    # Block normalization
    block_x, block_y = block_size
    num_blocks_x = num_cells_x - block_x + 1
    num_blocks_y = num_cells_y - block_y + 1
    normalized_blocks = np.zeros((num_blocks_y, num_blocks_x, block_y * block_x * nbins))

    for i in range(num_blocks_y):
        for j in range(num_blocks_x):
            block = histograms[i:i+block_y, j:j+block_x, :].ravel()
            normalized_blocks[i, j, :] = block / np.sqrt(np.sum(block**2) + 1e-6)

    hog_features = normalized_blocks.ravel()
    return hog_features

def load_data(coco_file, image_base_dir):
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)

    images = coco_data['images']
    annotations = coco_data['annotations']

    car_category_id = None
    for category in coco_data['categories']:
        if category['name'] == 'car':
            car_category_id = category['id']
            break

    if car_category_id is None:
        raise ValueError("car category ID not found in categories")

    car_annotations = [ann for ann in annotations if ann['category_id'] == car_category_id]
    image_id_to_annotations = {}
    for ann in car_annotations:
        if ann['image_id'] not in image_id_to_annotations:
            image_id_to_annotations[ann['image_id']] = []
        image_id_to_annotations[ann['image_id']].append(ann)

    image_paths = []
    for img in images:
        if img['id'] in image_id_to_annotations:
            image_paths.append((os.path.join(image_base_dir, img['file_name']), image_id_to_annotations[img['id']]))

    if not image_paths:
        raise ValueError("No image paths found for car annotations")

    return image_paths, car_category_id

def prepare_dataset(image_paths, car_category_id):
    X = []
    y = []

    for img_path, anns in image_paths:
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Warning: Unable to read image {img_path}")
            continue
        for ann in anns:
            bbox = ann['bbox']
            hog_features = extract_hog_features(image, bbox)
            X.append(hog_features)
            y.append(1)

    for img_path, anns in image_paths:
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        height, width = image.shape
        for _ in range(len(anns)):
            x = np.random.randint(0, width - 64)
            y_coord = np.random.randint(0, height - 128)
            hog_features = extract_hog_features(image, (x, y_coord, 64, 128))
            X.append(hog_features)
            y.append(0)

    X = np.array(X)
    y = np.array(y)
    return X, y


# # Paths to COCO files and image directories
# train_coco_file = '/images_thermal_train/coco.json'
# train_image_base_dir = '/images_thermal_train'
# test_coco_file = '/images_thermal_val/coco.json'
# test_image_base_dir = '/images_thermal_val'

# # Load and prepare the training dataset
# train_image_paths, train_car_category_id = load_data(train_coco_file, train_image_base_dir)
# X_train, y_train = prepare_dataset(train_image_paths, train_car_category_id)

# # Load and prepare the testing dataset
# test_image_paths, test_car_category_id = load_data(test_coco_file, test_image_base_dir)
# X_test, y_test = prepare_dataset(test_image_paths, test_car_category_id)



# # Train the SVM model
# svm = SVM()
# svm.fit(X_train, y_train)

# # Predict on the test set
# y_pred = svm.predict(X_test)


# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy: {accuracy * 100:.2f}%')

# # Classification report
# print(classification_report(y_test, y_pred))

def task1():
    # Paths to COCO files and image directories
    train_coco_file = ''
    train_image_base_dir = ''
    test_coco_file = ''
    test_image_base_dir = ''

    # Load and prepare the training dataset
    train_image_paths, train_car_category_id = load_data(train_coco_file, train_image_base_dir)
    X_train, y_train = prepare_dataset(train_image_paths, train_car_category_id)

    # Load and prepare the testing dataset
    test_image_paths, test_car_category_id = load_data(test_coco_file, test_image_base_dir)
    X_test, y_test = prepare_dataset(test_image_paths, test_car_category_id)



    # Train the SVM model
    svm = SVM()
    svm.fit(X_train, y_train)

    # Predict on the test set
    y_pred = svm.predict(X_test)


    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    # Classification report
    print(classification_report(y_test, y_pred))

