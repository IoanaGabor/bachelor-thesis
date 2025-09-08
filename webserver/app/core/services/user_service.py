from app.core.services.cognito_facade import CognitoFacade
from app.core.services.interfaces.user_service_interfaces import IUserService
from app.core.logger_config import logger

class UserService(IUserService):
    def __init__(self, cognito_facade: CognitoFacade):
        self._cognito = cognito_facade

    async def update_user_attributes(self, user_id: str, attributes: dict):
        return self._cognito.admin_update_user_attributes(user_id, attributes)

    async def get_user_attributes(self, sub: str):
        try:
            response = self._cognito.client.list_users(
                UserPoolId=self._cognito.user_pool_id,
                Filter=f'sub = "{sub}"'
            )
            logger.info(response)
            users = response.get('Users', [])
            if not users:
                return None
            user = users[0]
            attributes = {attr['Name']: attr['Value'] for attr in user.get('Attributes', [])}
            return {"email": attributes.get('email'), "custom:nsd_id": attributes.get('custom:nsd_id')}
        except Exception as e:
            return None

    async def get_nsd_id(self, sub: str):
        attributes = await self.get_user_attributes(sub)
        logger.info(attributes)
        if attributes is None:
            return None
        return attributes.get('custom:nsd_id')