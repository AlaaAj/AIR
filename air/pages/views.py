from django.shortcuts import render

from data.models import Data

# Create your views here.

def index(request):
    return render(request, 'pages/index.html', {'name':'ahmad'})


def search(request):
    return render(request, 'pages/search.html', {'name':'ahmad'})

def data(request): 
    data = Data.objects.all()
    return render(request , 'pages/data.html',{'data': data })
