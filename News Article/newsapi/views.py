from django.shortcuts import render

# Create your views here.
from rest_framework.views import APIView
from rest_framework.response import Response

from .serializers import PredictSerializer
from .ml.predict import predict_category

class PredictView(APIView):
    serializer_class = PredictSerializer 
    def get(self, request):
        return Response({
            "message": "Send a POST request with headline, short_description, and keywords to get prediction."
        })

    def post(self, request):
        ser = PredictSerializer(data=request.data)
        ser.is_valid(raise_exception=True)

        headline = ser.validated_data["headline"]
        description = ser.validated_data["short_description"]
        keywords = ser.validated_data["keywords"]

        category, confidence_scores = predict_category(
            headline, description, keywords
        )

        return Response({
            "predicted_category": category,
            "confidence_scores": confidence_scores
        })
