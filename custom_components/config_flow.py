import voluptuous as vol
from homeassistant import config_entries
from .const import DOMAIN  # Ensure this matches the domain of your integration

from .custompyephember.pyephember import EphEmber

class EphemberConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for EphEmber integration."""

    VERSION = 1

    async def async_step_user(self, user_input=None):
        """Handle the initial user step."""
        errors = {}

        if user_input is not None:
            username = user_input.get("username")
            password = user_input.get("password")

            # Validate user credentials (optional, you can implement custom validation)
            try:
                ember = await self.hass.async_add_executor_job(EphEmber, username, password)
                await self.hass.async_add_executor_job(ember.get_zones)
            except Exception as e:
                errors["base"] = "cannot_connect_EphEmber: " + repr(e)
            else:
                # Credentials valid, create entry
                return self.async_create_entry(title="EphEmber", data=user_input)

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema({
                vol.Required("username"): str,
                vol.Required("password"): str,
            }),
            errors=errors,
        )
