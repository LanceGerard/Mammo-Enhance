from django.http import HttpResponse
from django.shortcuts import render
from django.views.generic import TemplateView
from django.core.files.storage import FileSystemStorage
from .models import MammoEnhance as m
import os
from pathlib import Path
from urllib.parse import unquote

class Home(TemplateView):
    template_name = 'index.html'
    

def index(request):
    BASE_DIR = Path().resolve()
    context = {}

    if request.method == 'POST':
        # request file
        uploaded_file = request.FILES['selFile']

        # save file to /main/media/
        fs = FileSystemStorage()
        orig = fs.save(uploaded_file.name, uploaded_file)
        context['orig_url'] = fs.url(orig)
        #context['orig'] = fs.open(orig, mode='rb')

        # identify orig and export paths
        origPath = str(BASE_DIR)+unquote(fs.url(orig))
        exportPath = str(BASE_DIR)+"/main/media/"
        
        # set context for output
        context['proc_url'] = "/main/media/output.png"

        #print(unquote(fs.url(orig)))

        # get prediction
        prediction = m.processFile(origPath, exportPath)
        context['normal'] = ("%.2f" % (prediction[0]*100)).rstrip('0').rstrip('.')
        context['benign'] = ("%.2f" % (prediction[1]*100)).rstrip('0').rstrip('.')
        context['cancer'] = ("%.2f" % (prediction[2]*100)).rstrip('0').rstrip('.')
        #proc = fs.save("processed-"+uploaded_file.name,toProc)
        

    return render(request, "main/result.html", context)

