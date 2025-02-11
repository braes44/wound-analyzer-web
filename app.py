import os
from flask import Flask, request, render_template
import cv2
import numpy as np

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            return "No se ha subido ninguna imagen", 400
        
        file = request.files["image"]
        if file.filename == "":
            return "No se ha seleccionado ningún archivo", 400
        
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            pixel_to_cm = 0.026
            area_cm2 = area * (pixel_to_cm ** 2)
            perimeter_cm = perimeter * pixel_to_cm
            
            cv2.drawContours(img, [largest_contour], -1, (0, 255, 0), 2)
            cv2.putText(img, f'Area: {area_cm2:.2f} cm2', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, f'Perimetro: {perimeter_cm:.2f} cm', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            output_path = os.path.join("static", "processed_image.jpg")
            cv2.imwrite(output_path, img)
            
            return render_template("result.html", area=area_cm2, perimeter=perimeter_cm, image=output_path)
    
    return render_template("index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
