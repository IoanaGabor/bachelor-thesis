from abc import ABC, abstractmethod

class IUserService(ABC):
    @abstractmethod
    async def update_user_attributes(self, user_id: str, attributes: dict):
        pass

    @abstractmethod
    async def get_user_attributes(self, sub: str):
        pass

    @abstractmethod
    async def get_nsd_id(self, sub: str):
        pass