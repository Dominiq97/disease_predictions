from django.urls import path

from .views import (
    create_diagnosis,
    result,
    SymptomListView,
)

app_name = 'store'

urlpatterns = [

    path('symptom/create_diagnosis', create_diagnosis, name='create_diagnosis'),
    path('symptom/result', result, name='result'),
    path('symptom/list', SymptomListView.as_view(), name='symptom_list'),

]
