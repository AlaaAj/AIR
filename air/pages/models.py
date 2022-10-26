from multiprocessing.connection import answer_challenge
from django.db import models

class Data (models.Model):

    quastion = models.TextField()
    answer = models.TextField()
    lang = models.CharField(max_length=500)