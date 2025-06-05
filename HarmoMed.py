import os
from binary_HarmoMed_img import run_image_processing_pipeline
from werkzeug.datastructures import FileStorage

# if __name__ == "__main__":
    
#     test_file_paths = ["output.jpg", "output.jpg", "output.jpg"]
#     for path in test_file_paths:
#         if os.path.exists(path):
#             with open(path, "rb") as f:
#                 uploaded_file = FileStorage(stream=f, filename=path)
#                 run_image_processing_pipeline("wtest2.jpg", uploaded_file)

def HarmoMed_lir(target_img,reference_img):
    
    test_file_paths = target_img
    for path in test_file_paths:
        if os.path.exists(path):
            with open(path, "rb") as f:
                uploaded_file = FileStorage(stream=f, filename=path)
                run_image_processing_pipeline(reference_img, uploaded_file)