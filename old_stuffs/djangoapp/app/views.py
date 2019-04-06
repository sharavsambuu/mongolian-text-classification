from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import render, redirect
from .forms import MongolianTextForm
import xmlrpc.client

def index(request):
    form = MongolianTextForm()
    return render(request, 'home.html', {'form' : form})

def classify(request):
    if request.method == "POST":
        form = MongolianTextForm(request.POST)
        if form.is_valid():
            content = form.cleaned_data['content']
            with xmlrpc.client.ServerProxy("http://localhost:50001/") as proxy:
                news_class = proxy.predict_class_from_text(content)
            return render(request, 'classify.html', {'content' : content, 'news_class': news_class})
        return redirect('index')
    else:
        return redirect('index')
