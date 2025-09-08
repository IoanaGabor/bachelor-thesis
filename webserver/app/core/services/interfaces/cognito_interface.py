from abc import ABC, abstractmethod

class ICognitoFacade(ABC):
    @abstractmethod
    def list_users(self, limit=60, pagination_token=None):
        pass

    @abstractmethod
    def admin_update_user_attributes(self, username: str, attributes: dict):
        pass

    @abstractmethod
    def delete_user(self, username: str):
        pass

    @abstractmethod
    def create_user(self, username: str, attributes: dict):
        pass

    @abstractmethod
    def add_user_to_group(self, username: str, group_name: str = "admin"):
        pass