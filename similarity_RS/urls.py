from django.urls import path
from . import views

urlpatterns = [
    # path('', admin.site.urls),
    path('restaurants/<str:location_id>/', views.getSimilarRestaurants),
    path('hotels/<str:location_id>/', views.getSimilarHotels),
    path('restaurants/location/<int:separator>/', views.getRecommendedLocationRestaurants),

]
