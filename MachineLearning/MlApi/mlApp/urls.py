from django.urls import path
import mlApp.views as views
urlpatterns = [
    path("add/", views.api_add, name='api_add'),
    path("add_values/", views.add_val, name='api_add_values')
]
