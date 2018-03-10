from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import render, redirect
from .forms import MongolianTextForm

def index(request):
    form = MongolianTextForm()
    return render(request, 'home.html', {'form' : form})

def classify(request):
    if request.method == "POST":
        form = MongolianTextForm(request.POST)
        if form.is_valid():
            content = form.cleaned_data['content']
            return render(request, 'classify.html', {'content' : content})
        return redirect('index')
    else:
        return redirect('index')
