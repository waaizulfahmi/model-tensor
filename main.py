import os
import base64
import cv2
import numpy as np
import uuid
import requests
import shutil
from flask import Flask, request, jsonify
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import joblib
from keras_facenet import FaceNet

app = Flask(__name__)

# Constants
url = "https://nurz.site/api"
base_path = os.getcwd()
uploads_dir = "uploads"
models_dir = "models"
knn_model_file = "knn_model.pkl"
data_file = "data_baru.pkl"
THRESHOLD_KNN = 0.7

model = FaceNet()

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Ensure unique directories exist
for dir_name in [uploads_dir, models_dir]:
    os.makedirs(dir_name, exist_ok=True)


# Define a function to extract embeddings from an image using InceptionResnetV1
def get_embedding(img_path):
    # Read image
    img = cv2.imread(img_path)
    # Detect faces using Haar Cascade
    wajah = face_cascade.detectMultiScale(img, 1.1, 4)

    # If no face is detected, skip to the next image
    if len(wajah) == 0:
        return None

    # Extract face region
    x1, y1, width, height = wajah[0]
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    img = img[y1:y2, x1:x2]

    # Resize image to (160, 160)
    img = cv2.resize(img, (160, 160))

    # Convert image to tensor
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)

    # Prediksi kelas gambar
    # pred = model_anti_spoofing.predict(img)

    # Embed face using model
    embedding = model.embeddings(img)[0, :]

    return embedding


@app.route("/")
def hello():
    return "face recognition"


@app.route("/test")
def test():
    try:
        jb = joblib.load(base_path + "/models/knn_model.pkl")

        classs = jb.classes_.tolist()
        print(classs)
        return jsonify(classs), 200
    except Exception as e:
        # Handle exceptions here
        print("An error occurred:", str(e))
        # You can return an error response or log the error as needed.
        return jsonify({"error": str(e)}), 400


# Endpoint untuk registrasi
@app.route("/register", methods=["POST"])
def register_user():
    try:
        name = request.form.get("name")
        email = request.form.get("email")
        password = request.form.get("password")
        konfirmasi_password = request.form.get("konfirmasi_password")
        host = request.form.get("host")

        if name:
            first_name = name.split()[0]

        images = request.form.getlist("images[]")

        # Create a unique directory
        unique_directory_name = str(uuid.uuid4())[:6]
        unique_directory_path = os.path.join(
            "uploads", f"{first_name}-{unique_directory_name}"
        )

        # Ensure the unique directory exists (or is created)
        os.makedirs(unique_directory_path, exist_ok=True)

        # Process and save images
        for img in images:
            parts = img.split(",")
            base64_data = parts[1]
            mime_type = parts[0].split(";")[0].split(":")[1]
            extension = "png" if "/" in mime_type else mime_type.split("/")[1]

            # Generate a unique file name
            file_name = f"image_{uuid.uuid4()}.{extension}"

            # Decode data base64
            image_data_decoded = base64.b64decode(base64_data)

            # Save the file in the unique directory
            with open(
                os.path.join(unique_directory_path, file_name),
                "wb",
            ) as f:
                f.write(image_data_decoded)

        # dataset
        file_dir = f"{first_name}-{unique_directory_name}"
        root_dir = f"uploads/{file_dir}"

        # get list of image file paths and labels
        image_paths = []
        labels = []
        for subdir, dirs, files in os.walk(root_dir):
            label = os.path.basename(subdir)
            for file in files:
                if (
                    file.endswith(".jpg")
                    or file.endswith(".jpeg")
                    or file.endswith(".png")
                ):
                    image_paths.append(os.path.join(subdir, file))
                    labels.append(label)

        # Iterate through all images in image_paths
        new_embeddings = []
        new_labels = []
        for i, img_data in enumerate(image_paths):
            # Extract embedding from image and append to list
            new_test_embedding = get_embedding(img_data)

            # Check if embedding is None
            if new_test_embedding is None:
                continue

            new_embeddings.append(new_test_embedding)
            new_labels.append(labels[i])

        data = os.path.join(base_path, "models/data_baru.pkl")

        if os.path.exists(data):
            # Load model from file
            data_master = joblib.load(data)

            # Combine training data and labels
            data_master["embeddings"].extend(new_embeddings)
            data_master["labels"].extend(new_labels)

            try:
                joblib.dump(data_master, data)
                print("Data updated")
            except Exception as e:
                print("Error:", str(e))
        else:
            data_master = {"labels": new_labels, "embeddings": new_embeddings}
        try:
            joblib.dump(data_master, data)
            print("Data saved")
        except Exception as e:
            print("Error:", str(e))

        if len(new_embeddings) != 0:
            data = {
                "name": name,
                "email": email,
                "password": password,
                "konfirmasi_password": konfirmasi_password,
                "faceId": file_dir,
                "host": host,
            }

            response = requests.post(url + "/register", data=data)

            if response.status_code == 201:
                # Grid search parameters
                param_grid = {
                    "n_neighbors": [1, 3, 5, 7],
                    "weights": ["uniform", "distance"],
                    "metric": ["euclidean", "manhattan"],
                }
                # Initialize KNN and GridSearchCV
                knn = KNeighborsClassifier()
                grid_search = GridSearchCV(knn, param_grid, cv=5)
                # Perform grid search on your data
                grid_search.fit(data_master["embeddings"], data_master["labels"])
                # Print the best parameters and the best cross-validation score
                print("Best Parameters:", grid_search.best_params_)
                print("Best CV Score:", grid_search.best_score_)

                # Save the best model to a file
                best_model = grid_search.best_estimator_
                model_filename = os.path.join(base_path, "models/knn_model.pkl")
                joblib.dump(best_model, model_filename)
                print("Best model saved as", model_filename)
                return response.text, response.status_code
            else:
                # Request failed
                if os.path.exists(root_dir):
                    shutil.rmtree(root_dir)
                print("POST request failed with status code:", response.status_code)
                return response.text, response.status_code
        else:
            if os.path.exists(root_dir):
                shutil.rmtree(root_dir)
            return jsonify("tidak ada wajah"), 400

    except Exception as e:
        # Handle exceptions here
        print("An error occurred:", str(e))
        if os.path.exists(root_dir):
            shutil.rmtree(root_dir)
        # You can return an error response or log the error as needed.
        return jsonify({"error": str(e)}), 400


# Endpoint untuk login
@app.route("/login-face", methods=["POST"])
def login_user():
    image_path = ""
    try:
        knn_model_path = os.path.join(models_dir, knn_model_file)
        best_model_knn = joblib.load(knn_model_path)
        file = request.form.get("image")
        # If the user does not select a file, the browser may submit an empty part without a filename
        if file == None:
            return "No selected file", 400

        # Generate a unique filename for the uploaded image
        parts = file.split(",")
        base64_data = parts[1]
        mime_type = parts[0].split(";")[0].split(":")[1]
        extension = "png" if "/" in mime_type else mime_type.split("/")[1]

        # Generate a unique file name
        file_name = f"image_{str(uuid.uuid4())[:6]}.{extension}"

        # Decode data base64
        image_data_decoded = base64.b64decode(base64_data)

        # Save the file in the uploads directory
        image_path = os.path.join("uploads", file_name)
        with open(image_path, "wb") as f:
            f.write(image_data_decoded)

        def get_label(frame):
            vector = get_embedding(frame)

            if vector is None:
                label_knn = "Tidak Terdeteksi"
                score_knn = 0
            else:
                vector = vector.reshape(1, -1)
                y_pred_knn = best_model_knn.predict(vector)
                distances, _ = best_model_knn.kneighbors(vector)

                for i, pred_label in enumerate(y_pred_knn):
                    score_knn = distances[i][0]
                    if distances[i][0] > THRESHOLD_KNN:
                        label_knn = "Tidak Terdaftar"
                    else:
                        label_knn = pred_label

            return label_knn, score_knn

        label_knn, score_knn = get_label(image_path)

        faceId = label_knn

        # Remove the temporary image file
        if os.path.exists(image_path):
            os.remove(image_path)

        print(score_knn)
        print(faceId)
        if faceId not in ["Tidak Terdaftar", "Tidak Terdeteksi"]:
            data = {"faceId": faceId}
            response = requests.post(url + "/login-face", data=data)

            if response.status_code in [201, 200]:
                return response.text, response.status_code
            else:
                return jsonify("gagal login"), response.status_code
        else:
            return jsonify(faceId), 400
    except Exception as e:
        # Handle exceptions here
        # Remove the temporary image file
        if os.path.exists(image_path):
            os.remove(image_path)
        print("An error occurred:", str(e))
        return jsonify({"error": str(e)}), 400


@app.route("/refresh-model", methods=["GET"])
def refresh_model():
    face_dir = "uploads"
    image_paths = []
    labels = []

    # Collect image file paths and labels
    for subdir, _, files in os.walk(face_dir):
        label = os.path.basename(subdir)
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(subdir, file))
                labels.append(label)

    # Process and store new embeddings and labels
    new_embeddings = []
    new_labels = []

    for img_data in image_paths:
        new_test_embedding = get_embedding(img_data)

        if new_test_embedding is not None:
            new_embeddings.append(new_test_embedding)
            new_labels.append(labels[image_paths.index(img_data)])

    # Define the path for the data file
    data_path = os.path.join(base_path, "models/data_baru.pkl")

    # Create a dictionary with new labels and embeddings
    data_master = {"labels": new_labels, "embeddings": new_embeddings}

    try:
        # Save the data to the data file
        joblib.dump(data_master, data_path)
        print("Data saved")
    except Exception as e:
        print("Error:", str(e))

    # Define grid search parameters
    param_grid = {
        "n_neighbors": [1, 3, 5, 7],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"],
    }

    # Initialize KNN classifier and GridSearchCV
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=5)

    # Perform grid search on the data
    grid_search.fit(data_master["embeddings"], data_master["labels"])

    # Print the best parameters and the best cross-validation score
    print("Best Parameters:", grid_search.best_params_)
    print("Best CV Score:", grid_search.best_score_)

    # Save the best model to a file
    best_model = grid_search.best_estimator_
    model_filename = os.path.join(base_path, "models/knn_model.pkl")

    try:
        joblib.dump(best_model, model_filename)
        print("Best model saved as", model_filename)
        return "Success refresh"
    except Exception as e:
        print("Error:", str(e))
        return "Gagal refresh", 400


if __name__ == "__main__":
    server_port = os.environ.get("PORT", "8080")
    app.run(debug=True, port=server_port, host="127.0.0.1")
