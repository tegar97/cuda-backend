from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import zipfile
from io import BytesIO
import numpy as np
import cuda
import cv2
import os
import tempfile
import shutil
from fastapi.staticfiles import StaticFiles
import time

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    # Direktori sementara untuk menyimpan file
    UPLOAD_DIR = "uploads"
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # Validasi tipe file
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="File must be JPEG or PNG")
    
    # Baca konten file sebagai bytes
    file_bytes = await file.read()

    try:
        # Konversi byte ke array numpy
        np_array = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        # *Proses Gambar dengan OpenCV*
        # Contoh 1: Ubah ke grayscale
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Contoh 2: Resize gambar menjadi 256x256
        resized_image = cv2.resize(grayscale_image, (256, 256))

        # Simpan hasil proses ke buffer
        _, buffer = cv2.imencode(".png", resized_image)
        processed_image = BytesIO(buffer)

        # Kembalikan gambar hasil proses
        return StreamingResponse(processed_image, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "FastAPI Image Processing API"}

@app.post("/upload-zip/")
async def upload_zip(file: UploadFile = File(...), blur_parameter: float = 0.3):
    # Periksa apakah file adalah ZIP
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="File harus berformat ZIP.")

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Unzip dataset
            zip_path_input = os.path.join(temp_dir, file.filename)
            with open(zip_path_input, "wb") as buffer:
                buffer.write(await file.read())

            zip_path = os.path.join(temp_dir, "zip_temp")
            os.makedirs(zip_path, exist_ok=True)
            
            with zipfile.ZipFile(zip_path_input, 'r') as zip_ref:
                zip_ref.extractall(zip_path)

            # Load custom dataset
            images, labels, class_map = cuda.load_custom_dataset(zip_path)
            print(f"Loaded {len(images)} images from {len(class_map)} classes.")

            # Create filters for each class
            num_classes = len(class_map)
            kernel_size = 3
            center_parameter = 1.0

            filters = cuda.create_class_filters(num_classes, kernel_size, blur_parameter, center_parameter)
            print(f"Created filters for {num_classes} classes with blur parameter: {blur_parameter}")

            # Create directories for sample images
            samples_dir = "static/samples"
            os.makedirs(samples_dir, exist_ok=True)

            # Store sample results
            sample_results = []
            
            # Process only first image from each class as sample
            processed_classes = set()
            
            for i, (img, label) in enumerate(zip(images, labels)):
                class_name = list(class_map.keys())[label]
                
                # Skip if we already have a sample for this class
                if class_name in processed_classes:
                    continue
                    
                processed_classes.add(class_name)
                
                # Apply filter
                filtered_img = cuda.apply_class_filter(img, label, filters)
                
                # Save original and filtered images
                orig_filename = f"original_{class_name}.jpg"
                filtered_filename = f"filtered_{class_name}.jpg"
                
                orig_path = os.path.join(samples_dir, orig_filename)
                filtered_path = os.path.join(samples_dir, filtered_filename)
                
                cv2.imwrite(orig_path, img)
                cv2.imwrite(filtered_path, filtered_img)
                
                # Add to results
                sample_results.append({
                    "class_name": class_name,
                    "original_image": f"/static/samples/{orig_filename}",
                    "filtered_image": f"/static/samples/{filtered_filename}"
                })

            # Process all images and create zip files
            output_path = os.path.join(temp_dir, "filtered_dataset")
            os.makedirs(output_path, exist_ok=True)
            
            for i, img in enumerate(images):
                label = labels[i]
                class_name = list(class_map.keys())[label]
                filtered_img = cuda.apply_class_filter(img, label, filters)

                # Save the filtered image
                class_folder = os.path.join(output_path, class_name)
                os.makedirs(class_folder, exist_ok=True)
                output_file = os.path.join(class_folder, f"filtered_{i}.jpg")
                cv2.imwrite(output_file, filtered_img)

            # Create zip file and move to static directory
            downloads_dir = "static/downloads"
            os.makedirs(downloads_dir, exist_ok=True)
            
            zip_filename = f"filtered_results_{int(time.time())}.zip"
            result_zip = os.path.join(downloads_dir, zip_filename)
            shutil.make_archive(result_zip[:-4], 'zip', output_path)
            
            return {
                "samples": sample_results,
                "download_url": f"/static/downloads/{zip_filename}",
                "blur_parameter": blur_parameter
            }

    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="File ZIP tidak valid.")
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Terjadi kesalahan saat memproses file.")