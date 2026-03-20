# config.py
# แยก config และพารามิเตอร์สำหรับปรับแต่งการแข่งขัน

class Config:
    DEBUG = True
    MODEL_NAME = "indicator-j3riv"
    MODEL_VERSION = 4
    API_KEY = "6GfuPFK2Ue4Fvh9EZiQJ"
    UPLOAD_FOLDER = 'static/uploads/'
    RESULT_FOLDER = 'static/results/'
    ORANGE_STRENGTH = 1.15
    COLOR_TOLERANCE = 0.3
    MAX_ITER = 10
    PREPROCESS_SIZE = (256, 256)
    DELTAE_THRESHOLD = 10
    IOU_THRESHOLD = 0.5

# สามารถปรับแต่งค่าต่าง ๆ ได้จากไฟล์นี้
