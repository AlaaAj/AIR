from django.db import models

# Create your models here.

class Data (models.Model):

    quastion = models.TextField()
    answer = models.TextField()
    lang = models.CharField(max_length=500)