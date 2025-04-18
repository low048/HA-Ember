"""
Support for the EPH Controls Ember thermostats.
Forked from https://www.home-assistant.io/integrations/ephember
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any
import time
import logging

from .custompyephember.pyephember import (
    EphEmber,
    ZoneMode,
    zone_current_temperature,
    zone_is_active,
    zone_is_boost_active,
    zone_mode,
    zone_name,
    zone_target_temperature,
)
import voluptuous as vol

from homeassistant.components.climate import (
    PLATFORM_SCHEMA,
    ClimateEntity,
    ClimateEntityFeature,
    HVACAction,
    HVACMode,
)
from homeassistant.const import (
    ATTR_TEMPERATURE,
    CONF_PASSWORD,
    CONF_USERNAME,
    UnitOfTemperature,
)
from homeassistant.core import HomeAssistant
from homeassistant.config_entries import ConfigEntry
from homeassistant.helpers.entity_platform import AddEntitiesCallback
import homeassistant.helpers.config_validation as cv

_LOGGER = logging.getLogger(__name__)

SCAN_INTERVAL = timedelta(seconds=120)
OPERATION_LIST = [HVACMode.AUTO, HVACMode.HEAT, HVACMode.OFF]

PLATFORM_SCHEMA = PLATFORM_SCHEMA.extend({
    vol.Required(CONF_USERNAME): cv.string,
    vol.Required(CONF_PASSWORD): cv.string,
})

EPH_TO_HA_STATE = {
    "AUTO": HVACMode.AUTO,
    "ON": HVACMode.HEAT,
    "OFF": HVACMode.OFF,
}
HA_STATE_TO_EPH = {value: key for key, value in EPH_TO_HA_STATE.items()}


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback
) -> bool:
    """Set up EphEmber thermostats from a config entry."""
    username = entry.data.get(CONF_USERNAME)
    password = entry.data.get(CONF_PASSWORD)

    try:
        ember = EphEmber(username, password)
        if not await ember.async_login():
            _LOGGER.error("EphEmber login failed")
            return False

        # Retrieve a flattened list of zones from all homes.
        # Filter out any zone that does not have a valid deviceType.
        all_zones = await ember.async_get_zones()
        zones = [zone for zone in all_zones if zone.get("deviceType") is not None]
        if not zones:
            _LOGGER.error("No valid zones found from EphEmber. Available zones: %s",
                          await ember.get_zone_names())
            return False

        # Create a climate entity for each valid zone.
        entities = [EphEmberThermostat(ember, zone) for zone in zones]
        async_add_entities(entities)
    except Exception as e:
        _LOGGER.error("Cannot connect to EphEmber: %s", repr(e))
        return False

    return True


class EphEmberThermostat(ClimateEntity):
    """Representation of an EphEmber thermostat."""

    _attr_hvac_modes = OPERATION_LIST
    _attr_temperature_unit = UnitOfTemperature.CELSIUS

    def __init__(self, ember: EphEmber, zone: dict):
        """Initialize the thermostat entity."""
        self._ember = ember
        self._zone = zone
        self._zone_name = zone_name(zone)
        # Only valid zones have a deviceType. (This filters out e.g. 'Home'.)
        self._hot_water = zone.get("deviceType") == 4
        self._attr_name = self._zone_name

        if self._hot_water:
            self._attr_supported_features = (
                ClimateEntityFeature.TURN_ON | ClimateEntityFeature.TURN_OFF
            )
            self._attr_target_temperature_step = None
        else:
            self._attr_supported_features = (
                ClimateEntityFeature.TARGET_TEMPERATURE
                | ClimateEntityFeature.TURN_ON
                | ClimateEntityFeature.TURN_OFF
            )
            self._attr_target_temperature_step = 0.5

    @property
    def current_temperature(self) -> float:
        """Return the current temperature (defaulting to 0.0 if missing)."""
        temp = zone_current_temperature(self._zone)
        return temp if temp is not None else 0.0

    @property
    def target_temperature(self) -> float:
        """Return the target temperature (defaulting to 0.0 if missing)."""
        temp = zone_target_temperature(self._zone)
        return temp if temp is not None else 0.0

    @property
    def hvac_action(self) -> HVACAction:
        """Return the current HVAC action."""
        if zone_is_active(self._zone):
            return HVACAction.HEATING
        return HVACAction.IDLE

    @property
    def hvac_mode(self) -> HVACMode:
        """Return the current HVAC mode."""
        mode = zone_mode(self._zone)
        if mode is None:
            return HVACMode.OFF
        return self.map_mode_eph_hass(mode)

    async def async_set_hvac_mode(self, hvac_mode: HVACMode) -> None:
        """Set the HVAC mode asynchronously."""
        mode = self.map_mode_hass_eph(hvac_mode)
        if mode is not None:
            await self._ember.set_zone_mode(self._zone_name, mode)
        else:
            _LOGGER.error("Invalid HVAC mode: %s", hvac_mode)

    async def async_turn_on(self) -> None:
        """Turn on the thermostat (set to HEAT)."""
        await self.async_set_hvac_mode(HVACMode.HEAT)

    async def async_turn_off(self) -> None:
        """Turn off the thermostat."""
        await self.async_set_hvac_mode(HVACMode.OFF)

    @property
    def is_aux_heat(self) -> bool:
        """Return whether auxiliary (boost) heat is active."""
        return zone_is_boost_active(self._zone)

    async def async_turn_aux_heat_on(self) -> None:
        """Turn auxiliary heat on."""
        await self._ember.activate_zone_boost(self._zone_name, zone_target_temperature(self._zone))

    async def async_turn_aux_heat_off(self) -> None:
        """Turn auxiliary heat off."""
        await self._ember.deactivate_zone_boost(self._zone_name)

    async def async_set_temperature(self, **kwargs: Any) -> None:
        """Set a new target temperature asynchronously."""
        temperature = kwargs.get(ATTR_TEMPERATURE)
        if temperature is None:
            return
        if self._hot_water:
            return
        if temperature == self.target_temperature:
            return
        if temperature > self.max_temp or temperature < self.min_temp:
            return
        await self._ember.set_zone_target_temperature(self._zone_name, temperature)

    @property
    def min_temp(self) -> float:
        """Return the minimum allowed temperature."""
        if self._hot_water:
            return zone_target_temperature(self._zone) or 0.0
        return 5.0

    @property
    def max_temp(self) -> float:
        """Return the maximum allowed temperature."""
        if self._hot_water:
            return zone_target_temperature(self._zone) or 0.0
        return 30.0

    async def async_update(self) -> None:
        """Update the zone state asynchronously."""
        try:
            zone = await self._ember.get_zone(self._zone_name)
            if zone:
                self._zone = zone
            else:
                _LOGGER.warning("Zone '%s' not found during update", self._zone_name)
        except Exception as err:
            _LOGGER.error("Error updating zone '%s': %s", self._zone_name, err)

    @staticmethod
    def map_mode_hass_eph(operation_mode) -> str | None:
        """Map from Home Assistant HVAC mode to EPH mode."""
        return getattr(ZoneMode, HA_STATE_TO_EPH.get(operation_mode), None)

    @staticmethod
    def map_mode_eph_hass(operation_mode) -> HVACMode:
        """Map from EPH mode to Home Assistant HVAC mode."""
        if operation_mode is None:
            return HVACMode.OFF
        return EPH_TO_HA_STATE.get(operation_mode.name, HVACMode.HEAT_COOL)
