from django.contrib import admin
from django.urls import path, include
from django.views.generic.base import TemplateView

urlpatterns = [
    path('', TemplateView.as_view(template_name='home.html'), name='home'),
    path('', include('app.urls')),
    path('', include('django.contrib.auth.urls')),
    path('admin/', admin.site.urls),
]
