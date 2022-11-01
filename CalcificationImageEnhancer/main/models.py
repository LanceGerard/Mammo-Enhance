from django.db import models
from .utils import *
from PIL import Image
import numpy as np
from io import BytesIO
from django.core.files.base import ContentFile
from tkinter import *

# Create your models here.
ACTION_CHOICES= (
    ('PROCESS', 'process'),
    ('NON', 'non')
)

class MammoEnhance(models.Model):
    image = models.ImageField(upload_to='images')
    action = models.CharField(max_length=50, choices=ACTION_CHOICES)
    updated = models.DateTimeField(auto_now=True)
    created = models.DateTimeField(auto_now_add=True)

    

    orig_img = ""

    def __str__(self):
        return str(self.id)

    # PROCESS BTN FUNC
    def processFile(filename, path):
        raw_img = cv2.imread(filename)
        
        # greyed
        gray1_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)

        # contrast
        contrasted_img = cv2.convertScaleAbs(gray1_img, alpha=1.0, beta=0)

        # filterings
        kernel1 = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) # sharpen kernel
        # kernel2 = np.array([[0,1,0], [1,-4,1], [0,1,0]]) # laplacian sharpen kernel
        kernel3 = np.array([[0.075, 0.124, 0.075],[0.124, 0.204, 0.124],[0.075, 0.124, 0.075]]) # gaussian kernel
        gaussed_img = cv2.filter2D(contrasted_img, -1, kernel3)
        filtered_img = cv2.filter2D(gaussed_img, -1, kernel1)

        #greyed
        #gray2_img = cv2.cvtColor(arranged_img1, cv2.COLOR_RGB2GRAY)

        #apply clahe
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        equalized_img = clahe.apply(filtered_img)
        
        result_img = cv2.cvtColor(equalized_img, cv2.COLOR_GRAY2RGB)

        #Rearrange colors
        #arranged_img = rearrange(raw_img)

        #Resize
        #resized_img = resize(arranged_img)

        # Write
        cv2.imwrite(os.path.join(path, "output.png"), equalized_img)

        # predict
        prediction = predict(result_img)
        return prediction
    
