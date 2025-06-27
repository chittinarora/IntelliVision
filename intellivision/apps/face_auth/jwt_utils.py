"""
jwt_utils.py - Face Auth App
Utility for generating JWT tokens for users using SimpleJWT.
"""

from rest_framework_simplejwt.tokens import RefreshToken

def get_tokens_for_user(user):
    """
    Generate a JWT access token for the given user.
    """
    refresh = RefreshToken.for_user(user)
    return str(refresh.access_token)
