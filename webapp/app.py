import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from car_model_predictor import predict_car_model, class_labels


app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Handle file upload
        image = request.files["image"]
        filename = secure_filename(image.filename)
        image_path = os.path.join("uploads", filename)
        image.save(image_path)

        # Load the model and make predictions
        predicted_class, confidence = predict_car_model(image_path, class_labels)
        os.remove(image_path)  # Clean up the uploaded file

        result = f"{predicted_class}, Confidence: {confidence}"
        
        return render_template("result.html", result=result)
    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
