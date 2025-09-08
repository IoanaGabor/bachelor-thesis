class UserDTO:
    def __init__(self, user_id, username, enabled, user_status, email, full_name, disabled, groups, attributes):
        self.user_id = user_id
        self.username = username
        self.enabled = enabled
        self.user_status = user_status
        self.email = email
        self.full_name = full_name
        self.disabled = disabled
        self.groups = groups
        self.attributes = attributes

    def to_dict(self):
        return {
            "user_id": self.user_id,
            "username": self.username,
            "enabled": self.enabled,
            "user_status": self.user_status,
            "email": self.email,
            "full_name": self.full_name,
            "disabled": self.disabled,
            "groups": self.groups,
            "attributes": self.attributes
        }
