from django import forms
from django.forms import (formset_factory, modelformset_factory)

from .models import (Symptom)


class SymptomForm(forms.Form):
    name = forms.CharField(
        label='Symptom: ',
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Ex: anxiety'
        })
    )

SymptomFormset = formset_factory(SymptomForm)

