from django.urls import path
from . import views

urlpatterns = [
    path('', views.analisis_completo, name='home'),

    # NUEVO ENDPOINT API
    path('api/predict/', views.api_predict, name='api_predict'),
]
