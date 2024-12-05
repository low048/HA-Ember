import voluptuous as vol
from homeassistant import config_entries
from .const import DOMAIN  # Ensure this matches the domain of your integration

from .custompyephember.pyephember import EphEmber

import logging
_LOGGER = logging.getLogger(__name__)

class EphemberConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for EphEmber integration."""

    VERSION = 1

    async def async_step_user(self, user_input=None):
        """Handle the initial user step."""
        errors = {}

        if user_input is not None:
            username = user_input.get("username")
            password = user_input.get("password")

            # Validate user credentials (asynchronous call)
            try:
                # Instantiate EphEmber
                ember = EphEmber(username, password)

                # Call async_login directly
                login_success = await ember.async_login()

                # Validate login success
                if not login_success:
                    errors["base"] = "invalid_credentials"
                else:
                    # Test fetching zones to ensure setup is complete
                    zones = await ember.async_get_zones()
                    if not zones:
                        errors["base"] = "no_zones_found"
                    else:
                        # Credentials valid, create entry
                        return self.async_create_entry(title="EphEmber", data=user_input)

            except Exception as e:
                errors["base"] = "cannot_connect"
                _LOGGER.error(f"Error during configuration flow: {repr(e)}")

        # Show the form again if there were errors
        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema({
                vol.Required("username"): str,
                vol.Required("password"): str,
            }),
            errors=errors,
        )
    