from rest_framework import serializers

class PredictSerializer(serializers.Serializer):
    headline = serializers.CharField()
    short_description = serializers.CharField()
    keywords = serializers.CharField()
