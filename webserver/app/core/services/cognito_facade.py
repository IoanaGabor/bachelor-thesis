import boto3
from app.core.services.interfaces.cognito_interface import ICognitoFacade

class CognitoFacade(ICognitoFacade):
    def __init__(self, user_pool_id: str, region_name: str, aws_access_key_id: str, aws_secret_access_key: str):
        self.client = boto3.client(
            'cognito-idp',
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
        self.user_pool_id = user_pool_id

    def list_users(self, limit=60, pagination_token=None):
        params = {
            'UserPoolId': self.user_pool_id,
            'Limit': limit
        }
        if pagination_token:
            params['PaginationToken'] = pagination_token

        response = self.client.list_users(**params)
        return response

    def admin_update_user_attributes(self, username: str, attributes: dict):
        user_attributes = [{'Name': k, 'Value': v} for k, v in attributes.items()]
        response = self.client.admin_update_user_attributes(
            UserPoolId=self.user_pool_id,
            Username=username,
            UserAttributes=user_attributes
        )
        return response

    def delete_user(self, username: str):
        response = self.client.admin_delete_user(
            UserPoolId=self.user_pool_id,
            Username=username
        )
        return response

    def create_user(self, username: str, attributes: dict):
        user_attributes = [{'Name': k, 'Value': v} for k, v in attributes.items()]
        response = self.client.admin_create_user(
            UserPoolId=self.user_pool_id,
            Username=username,
            UserAttributes=user_attributes
        )
        return response

    def add_user_to_group(self, username: str, group_name: str = "admin"):
        response = self.client.admin_add_user_to_group(
            UserPoolId=self.user_pool_id,
            Username=username,
            GroupName=group_name
        )
        return response
