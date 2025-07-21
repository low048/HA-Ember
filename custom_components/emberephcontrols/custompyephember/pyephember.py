"""
PyEphEmber interface implementation for https://ember.ephcontrols.com/
"""
# pylint: disable=consider-using-f-string

import base64
import datetime
import json
import time
import collections
import logging

from enum import Enum
from typing import OrderedDict

import aiohttp  # Use aiohttp for asynchronous HTTP calls
import paho.mqtt.client as mqtt

_LOGGER = logging.getLogger(__name__)


class ZoneMode(Enum):
    """
    Modes that a zone can be set to.
    """
    AUTO = 0
    ALL_DAY = 1
    ON = 2
    OFF = 3


def GetPointIndex(zone, pointIndex) -> int:
    assert isinstance(pointIndex, PointIndex)
    try:
        device_type = zone["deviceType"]
    except Exception:
        raise RuntimeError("Zone data missing 'deviceType'")
    match pointIndex:
        case PointIndex.ADVANCE_ACTIVE:
            return 4
        case PointIndex.CURRENT_TEMP:
            return 5
        case PointIndex.TARGET_TEMP:
            match device_type:
                case 773:
                    return 12
                case _:
                    return 6
        case PointIndex.MODE:
            match device_type:
                case 514 | 773:
                    return 11
                case 2 | 4:
                    return 7
                case _:
                    return 7
        case PointIndex.BOOST_HOURS:
            match device_type:
                case 514 | 773:
                    return 13
                case _:
                    return 8
        case PointIndex.BOOST_TIME:
            return 9
        case PointIndex.BOILER_STATE:
            return 10
        case PointIndex.BOOST_TEMP:
            return 14
        case PointIndex.CTR_15_ABAB:
            return 15
        case PointIndex.XXX_16_0000:
            return 16
        case PointIndex.CTR_17_ABAB:
            return 17
        case PointIndex.CTR_18_0AB7:
            return 18
        case _:
            raise RuntimeError("Unknown PointIndex:" + str(pointIndex))


class PointIndex(Enum):
    """
    Point indices for pointData returned by API.
    """
    ADVANCE_ACTIVE = 4
    CURRENT_TEMP = 5
    TARGET_TEMP = 6
    MODE = 7
    BOOST_HOURS = 8
    BOOST_TIME = 9
    BOILER_STATE = 10
    BOOST_TEMP = 14
    CTR_15_ABAB = 15
    XXX_16_0000 = 16
    CTR_17_ABAB = 17
    CTR_18_0AB7 = 18


# Named tuple to hold a command to write data to a zone.
ZoneCommand = collections.namedtuple("ZoneCommand", ["name", "value", "index"])


def zone_command_to_ints(zone, command):
    """
    Convert a ZoneCommand to an array of integers to send.
    """
    type_data = {
        "SMALL_INT": {"id": 1, "byte_len": 1},
        "TEMP_RO": {"id": 2, "byte_len": 2},
        "TEMP_RW": {"id": 4, "byte_len": 2},
        "TIMESTAMP": {"id": 5, "byte_len": 4},
    }
    writable_command_types = {
        "ADVANCE_ACTIVE": "SMALL_INT",
        "TARGET_TEMP": "TEMP_RW",
        "MODE": "SMALL_INT",
        "BOOST_HOURS": "SMALL_INT",
        "BOOST_TIME": "TIMESTAMP",
        "BOOST_TEMP": "TEMP_RW",
    }
    if command.name not in writable_command_types:
        raise ValueError("Cannot write to read-only value {}".format(command.name))
    command_type = writable_command_types[command.name]
    if command.index is not None:
        command_index = command.index
    else:
        command_index = GetPointIndex(zone, PointIndex[command.name])
    int_array = [0, command_index, type_data[command_type]["id"]]
    send_value = command.value
    if command_type == "TEMP_RW":
        send_value = int(10 * send_value)
    elif command_type == "TIMESTAMP":
        if isinstance(command.value, datetime.datetime):
            send_value = int(command.value.timestamp())
    for byte_value in send_value.to_bytes(type_data[command_type]["byte_len"], "big"):
        int_array.append(int(byte_value))
    return int_array


def zone_is_active(zone):
    """
    Check if the zone is active.
    """
    return (zone_boost_hours(zone) or 0) > 0 or zone_advance_active(zone)


def zone_advance_active(zone):
    """
    Check whether the zone's advance is active.
    """
    match zone["deviceType"]:
        case 773:
            return False
        case 514:
            return False
        case _:
            return (zone_pointdata_value(zone, PointIndex.ADVANCE_ACTIVE) or 0) != 0


def boiler_state(zone):
    """
    Return the boiler state for the zone.
    """
    return zone_pointdata_value(zone, PointIndex.BOILER_STATE)


def lastKey(dict_obj):
    return list(dict_obj.keys())[-1]


def firstKey(dict_obj):
    return list(dict_obj.keys())[0]


def try_parse_int(value):
    try:
        return int(value), True
    except ValueError:
        return None, False


def scheduletime_to_time(dict_obj, key_name):
    """
    Convert a schedule start/end time to a Python time.
    """
    if dict_obj.get(key_name) is None:
        return None
    stime = dict_obj[key_name]
    if stime is None:
        return None
    return datetime.time(int(str(stime)[:-1]), 10 * int(str(stime)[-1:]))


def getZoneTime(zone):
    """
    Return [time, weekday] for a zone.
    If 'timestamp' is missing, uses the current time.
    """
    ts = zone.get("timestamp")
    if ts is None:
        _LOGGER.warning("Zone '%s' missing 'timestamp'; using current time.", zone.get("name", "unknown"))
        ts = int(time.time() * 1000)
    tstamp = time.gmtime(ts / 1000)
    ts_time = datetime.time(tstamp.tm_hour, tstamp.tm_min)
    ts_wday = tstamp.tm_wday + 1
    if ts_wday == 7:
        ts_wday = 0
    return [ts_time, ts_wday]


def zone_get_running_day(zone):
    return zone["days"][getZoneTime(zone)[1]]


def zone_get_running_program(zone):
    mode = zone_mode(zone)
    ts_time = getZoneTime(zone)[0]
    todaysDay = zone_get_running_day(zone)
    if todaysDay is None:
        return None
    if mode == ZoneMode.AUTO:
        for key in todaysDay["programs"]:
            program = todaysDay["programs"][key]
            start_time = scheduletime_to_time(program, "startTime")
            end_time = scheduletime_to_time(program, "endTime")
            p_time = scheduletime_to_time(program, "time")
            if start_time is not None and end_time is not None and start_time <= ts_time <= end_time:
                return program
            elif p_time is not None and p_time >= ts_time:
                return [program.get("Prev"), program]
        lastProg = todaysDay["programs"][lastKey(todaysDay["programs"])]
        if lastProg.get("time") is None:
            return lastProg
        else:
            return [lastProg, lastProg.get("Next")]
    elif mode == ZoneMode.ALL_DAY:
        startProgram = todaysDay["programs"][firstKey(todaysDay["programs"])]
        endProgram = todaysDay["programs"][lastKey(todaysDay["programs"])]
        return [startProgram, endProgram]
    return None


def zone_is_scheduled_on(zone):
    mode = zone_mode(zone)
    if mode == ZoneMode.OFF:
        return False
    if mode == ZoneMode.ON:
        return True
    ts_time = getZoneTime(zone)[0]
    if mode == ZoneMode.AUTO:
        runningPrograms = zone_get_running_program(zone)
        if runningPrograms is None:
            return False
        elif isinstance(runningPrograms, list):
            currentTemp = zone_current_temperature(zone) or 0
            targetTemp = (runningPrograms[0].get("temperature") or 0) / 10.0
            return currentTemp + 0.3 < targetTemp
        else:
            start_time = scheduletime_to_time(runningPrograms, "startTime")
            end_time = scheduletime_to_time(runningPrograms, "endTime")
            if start_time is not None and end_time is not None and start_time <= ts_time <= end_time:
                return True
    elif mode == ZoneMode.ALL_DAY:
        runningPrograms = zone_get_running_program(zone)
        first_start_time = scheduletime_to_time(runningPrograms[0], "startTime")
        last_end_time = scheduletime_to_time(runningPrograms[1], "endTime")
        if first_start_time is None or last_end_time is None:
            return False
        return first_start_time <= ts_time <= last_end_time
    return False


def zone_name(zone):
    return zone.get("name", "Unknown")


def zone_is_boost_active(zone):
    return (zone_boost_hours(zone) or 0) > 0


def zone_boost_hours(zone):
    return zone_pointdata_value(zone, PointIndex.BOOST_HOURS) or 0


def zone_boost_timestamp(zone):
    return zone_pointdata_value(zone, PointIndex.BOOST_TIME) or 0


def zone_temperature(zone, label):
    """
    Return the temperature as a float (divided by 10).
    If the value is missing, return 0.0.
    """
    value = None
    if zone.get("deviceType") == 773:
        if zone_mode(zone) == ZoneMode.AUTO and label == PointIndex.TARGET_TEMP:
            programs = zone_get_running_program(zone)
            if programs is not None and programs[0].get("temperature") is not None:
                value = programs[0]["temperature"]
        else:
            temp_val = zone_pointdata_value(zone, PointIndex(label))
            if temp_val is not None:
                value = temp_val
    else:
        temp_val = zone_pointdata_value(zone, PointIndex(label))
        if temp_val is not None:
            value = temp_val
    if value is None:
        return 0.0
    return value / 10.0


def zone_target_temperature(zone):
    return zone_temperature(zone, PointIndex.TARGET_TEMP)


def zone_boost_temperature(zone):
    return zone_temperature(zone, PointIndex.BOOST_TEMP)


def zone_current_temperature(zone):
    return zone_temperature(zone, PointIndex.CURRENT_TEMP)


def zone_pointdata_value(zone, pointIndex):
    index = GetPointIndex(zone, pointIndex)
    pointDataList = zone.get("pointDataList", [])
    for datum in pointDataList:
        if datum.get("pointIndex") == index:
            try:
                return int(datum.get("value"))
            except Exception:
                return None
    return None


def zone_mode(zone):
    modeValue = zone_pointdata_value(zone, PointIndex.MODE)
    match modeValue:
        case 0:
            return ZoneMode.AUTO
        case 1 | 9:
            match zone.get("deviceType"):
                case 773:
                    return ZoneMode.ON
                case _:
                    return ZoneMode.ALL_DAY
        case 2 | 10:
            return ZoneMode.ON
        case 3 | 4:
            return ZoneMode.OFF
        case _:
            return None


def get_zone_mode_value(zone, mode) -> int:
    if mode == ZoneMode.AUTO:
        return 0
    match zone.get("deviceType"):
        case 773:
            match mode:
                case ZoneMode.ON:
                    return 1
                case ZoneMode.OFF:
                    return 4
        case 514:
            match mode:
                case ZoneMode.ALL_DAY:
                    return 9
                case ZoneMode.ON:
                    return 10
                case ZoneMode.OFF:
                    return 4
        case _:
            match mode:
                case ZoneMode.ALL_DAY:
                    return 1
                case ZoneMode.ON:
                    return 2
                case ZoneMode.OFF:
                    return 3


class EphMessenger:
    """
    MQTT interface to the EphEmber API.
    """
    def _zone_command_b64(self, zone, cmd, stop_mqtt=True, timeout=1):
        product_id = zone["productId"]
        uid = zone["uid"]
        msg = json.dumps({
            "common": {
                "serial": 7870,
                "productId": product_id,
                "uid": uid,
                "timestamp": str(int(1000 * time.time()))
            },
            "data": {
                "mac": zone["mac"],
                "pointData": cmd
            }
        })
        started_locally = False
        if not self.client or not self.client.is_connected():
            started_locally = True
            self.start()
        pub = self.client.publish("/".join([product_id, uid, "download/pointdata"]), msg, 0)
        pub.wait_for_publish(timeout=timeout)
        if started_locally and stop_mqtt:
            self.stop()
        return pub.is_published()

    def start(self, callbacks=None, loop_start=False):
        # Instead of calling an async function, use cached credentials.
        if not self.parent._login_data or "data" not in self.parent._login_data:
            raise RuntimeError("No valid login data cached! Ensure async_login() has been called and completed.")
        credentials = self.parent._login_data["data"]
        # Use the already cached user_id; if not, fallback to username.
        user_id = self.parent._user.get("user_id") or self.parent._user["username"]
        self.client_id = "{}_{}".format(user_id, str(int(1000 * time.time())))
        token = credentials["token"]
        mclient = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, self.client_id)
        mclient.tls_set()
        self.client = mclient
        user_name = "app/{}".format(token)
        mclient.username_pw_set(user_name, token)
        if callbacks is not None:
            for key in callbacks.keys():
                setattr(mclient, key, callbacks[key])
        mclient.connect(self.api_url, self.api_port)
        if loop_start:
            mclient.loop_start()
        return mclient

    def stop(self):
        if not self.client:
            return False
        if self.client.is_connected():
            self.client.disconnect()
        return True

    def send_zone_commands(self, zone, commands, stop_mqtt=True, timeout=1):
        def ints_to_b64_cmd(int_array):
            return base64.b64encode(bytes(int_array)).decode("ascii")
        if isinstance(commands, ZoneCommand):
            commands = [commands]
        ints_cmd = [x for cmd in commands for x in zone_command_to_ints(zone, cmd)]
        return self._zone_command_b64(zone, ints_to_b64_cmd(ints_cmd), stop_mqtt, timeout)

    def __init__(self, parent):
        self.api_url = "eu-base-mqtt.topband-cloud.com"
        self.api_port = 18883
        self.client = None
        self.client_id = None
        self.parent = parent


class EphEmber:
    """
    Interacts with a EphEmber thermostat via the API.
    """
    async def _http(self, endpoint, *, method="POST", headers=None, send_token=False, data=None, timeout=10):
        if not headers:
            headers = {}
        if send_token:
            if not await self._do_auth():
                raise RuntimeError("Unable to login")
            headers["Authorization"] = self._login_data["data"]["token"]
        headers["Content-Type"] = "application/json"
        headers["Accept"] = "application/json"
        url = f"{self.http_api_base}{endpoint}"
        async with aiohttp.ClientSession() as session:
            if method.upper() == "POST":
                async with session.post(url, json=data, headers=headers, timeout=timeout) as response:
                    if response.status != 200:
                        raise RuntimeError(f"{response.status} response code")
                    return await response.json()
            elif method.upper() == "GET":
                async with session.get(url, params=data, headers=headers, timeout=timeout) as response:
                    if response.status != 200:
                        raise RuntimeError(f"{response.status} response code")
                    return await response.json()
            else:
                raise ValueError("Unsupported HTTP method")

    def _requires_refresh_token(self):
        expires_on = self._login_data["last_refresh"] + datetime.timedelta(seconds=self._refresh_token_validity_seconds)
        refresh = datetime.datetime.utcnow() + datetime.timedelta(seconds=30)
        return expires_on < refresh

    async def _request_token(self, force=False):
        if self._login_data is None:
            raise RuntimeError("Don't have a token to refresh")
        if not force:
            if not self._requires_refresh_token():
                return True
        response = await self._http("appLogin/refreshAccessToken", method="GET", headers={'Authorization': self._login_data['data']['refresh_token']})
        refresh_data = response
        if "token" not in refresh_data.get("data", {}):
            return False
        self._login_data["data"] = refresh_data["data"]
        self._login_data["last_refresh"] = datetime.datetime.utcnow()
        return True

    async def _login(self):
        self._login_data = None
        response = await self._http("appLogin/login", data={'userName': self._user['username'], 'password': self._user['password']})
        self._login_data = response
        if self._login_data["status"] != 0:
            self._login_data = None
            return False
        self._login_data["last_refresh"] = datetime.datetime.utcnow()
        return "data" in self._login_data and "token" in self._login_data["data"]

    async def _do_auth(self):
        if self._login_data is None:
            return await self._login()
        return await self._request_token()

    async def _get_user_details(self):
        response = await self._http("user/selectUser", method="GET", send_token=True)
        user_details = response
        if user_details["status"] != 0:
            return {}
        return user_details

    async def _get_user_id(self, force=False):
        if not force and self._user.get("user_id"):
            return self._user["user_id"]
        user_details = await self._get_user_details()
        data = user_details.get("data", {})
        if "id" not in data:
            raise RuntimeError("Cannot get user ID")
        self._user["user_id"] = str(data["id"])
        return self._user["user_id"]

    def _get_first_gateway_id(self):
        if not self._homes:
            raise RuntimeError("Cannot get gateway id from list of homes.")
        return self._homes[0]["gatewayid"]

    def _set_zone_target_temperature(self, zone, target_temperature):
        return self.messenger.send_zone_commands(zone, ZoneCommand("TARGET_TEMP", target_temperature, GetPointIndex(zone, PointIndex.TARGET_TEMP)))

    def _set_zone_boost_temperature(self, zone, target_temperature):
        return self.messenger.send_zone_commands(zone, ZoneCommand("BOOST_TEMP", target_temperature, None))

    def _set_zone_advance(self, zone, advance=True):
        return self.messenger.send_zone_commands(zone, ZoneCommand("ADVANCE_ACTIVE", 1 if advance else 0, None))

    def _set_zone_boost(self, zone, boost_temperature, num_hours, timestamp=0):
        cmds = [ZoneCommand("BOOST_HOURS", num_hours, None)]
        if boost_temperature is not None:
            cmds.append(ZoneCommand("BOOST_TEMP", boost_temperature, None))
        if timestamp is not None:
            if timestamp == 0:
                timestamp = int(datetime.datetime.now().timestamp())
            cmds.append(ZoneCommand("BOOST_TIME", timestamp, None))
        return self.messenger.send_zone_commands(zone, cmds)

    def _set_zone_mode(self, zone, mode_num, index):
        return self.messenger.send_zone_commands(zone, ZoneCommand("MODE", mode_num, index))

    async def messenging_credentials(self):
        if not await self._do_auth():
            raise RuntimeError("Unable to login")
        return {"user_id": await self._get_user_id(), "token": self._login_data["data"]["token"]}

    async def list_homes(self):
        response = await self._http("homes/list", method="GET", send_token=True)
        homes = response
        status = homes.get("status", 1)
        if status != 0:
            raise RuntimeError("Error getting home: {}".format(status))
        return homes.get("data", [])

    async def get_home_details(self, gateway_id=None, force=False):
        if self._home_details and not force:
            return self._home_details
        if gateway_id is None:
            if not self._homes:
                self._homes = await self.list_homes()
            gateway_id = self._get_first_gateway_id()
        response = await self._http("homes/detail", send_token=True, data={"gateWayId": gateway_id})
        home_details = response
        status = home_details.get("status", 1)
        if status != 0:
            raise RuntimeError("Error getting details from home: {}".format(status))
        if "data" not in home_details or "homes" not in home_details["data"]:
            raise RuntimeError("Error getting details from home: no home data found")
        self._home_details = home_details["data"]
        return home_details["data"]

    @staticmethod
    def lastKey(dict_obj):
        return list(dict_obj.keys())[-1]

    @staticmethod
    def firstKey(dict_obj):
        return list(dict_obj.keys())[0]

    async def get_homes(self):
        homes_data = await self.list_homes()
        for home in homes_data:
            home["zones"] = []
            gateway_id = home["gatewayid"]
            response = await self._http("homesVT/zoneProgram", send_token=True, data={"gateWayId": gateway_id})
            homezones = response
            status = homezones.get("status", 1)
            if status != 0:
                raise RuntimeError("Error getting zones from home: {}".format(status))
            if "data" not in homezones:
                raise RuntimeError("Error getting zones from home: no data found")
            if "timestamp" not in homezones:
                raise RuntimeError("Error getting zones from home: no timestamp found")
            for zone in homezones["data"]:
                zone["days"] = {}
                prevProgramm = None
                for day in sorted(zone["deviceDays"], key=lambda x: x["dayType"]):
                    day["programs"] = {}
                    for key in day.keys():
                        if key.startswith("p"):
                            tryGetId = try_parse_int(key[1:])
                            if tryGetId[1]:
                                programm = day[key]
                                if programm is not None:
                                    if prevProgramm is not None:
                                        programm["Prev"] = prevProgramm
                                    programm["Count"] = tryGetId[0]
                                    prevProgramm = programm
                                    day["programs"][tryGetId[0]] = programm
                    zone["days"][day["dayType"]] = day
                lastProgramm = None
                firstProgramm = None
                for day in OrderedDict(sorted(zone["days"].items(), reverse=True)):
                    if lastProgramm is not None:
                        firstProgramm = zone["days"][day]["programs"][EphEmber.lastKey(zone["days"][day]["programs"])]
                        lastProgramm["Prev"] = firstProgramm
                    lastProgramm = zone["days"][day]["programs"][EphEmber.firstKey(zone["days"][day]["programs"])]
                lastProgramm["Prev"] = firstProgramm
                firstDayPrograms = zone["days"][EphEmber.firstKey(zone["days"])]["programs"]
                firstProgram = firstDayPrograms[EphEmber.firstKey(firstDayPrograms)]
                nextProgram = firstProgram
                for day in OrderedDict(sorted(zone["days"].items(), reverse=True)):
                    orderedProgs = OrderedDict(sorted(zone["days"][day]["programs"].items(), reverse=True))
                    for progNum in orderedProgs:
                        program = zone["days"][day]["programs"][progNum]
                        program["Next"] = nextProgram
                        nextProgram = program
                zone["timestamp"] = homezones["timestamp"]
                home["zones"].append(zone)
        return homes_data

    async def get_zones(self):
        """Return a flattened list of zones from all homes."""
        homes = await self.get_homes()
        zones = []
        for home in homes:
            zones.extend(home.get("zones", []))
        return zones

    async def async_get_zones(self):
        return await self.get_zones()

    async def get_zone_names(self):
        zone_names = []
        zones = await self.get_zones()
        for zone in zones:
            zone_names.append(zone.get("name", "Unknown"))
        return zone_names

    async def get_zone(self, zoneid):
        """
        Retrieve a zone by matching zoneid or zone name (case-insensitive).
        """
        zones = await self.get_zones()
        for zone in zones:
            zid = str(zone.get("zoneid", "")).lower()
            zname = str(zone.get("name", "")).lower()
            if str(zoneid).lower() in [zid, zname]:
                return zone
        _LOGGER.warning("Unknown zone: %s; available zones: %s", zoneid, await self.get_zone_names())
        return None

    async def is_zone_active(self, zoneid):
        zone = await self.get_zone(zoneid)
        if zone is None:
            return False
        return zone_is_active(zone)

    async def is_zone_boiler_on(self, zoneid):
        zone = await self.get_zone(zoneid)
        if zone is None:
            return False
        return boiler_state(zone) == 2

    async def get_zone_temperature(self, zoneid):
        zone = await self.get_zone(zoneid)
        if zone is None:
            return 0.0
        return zone_current_temperature(zone) or 0.0

    async def get_zone_target_temperature(self, zoneid):
        zone = await self.get_zone(zoneid)
        if zone is None:
            return 0.0
        return zone_target_temperature(zone) or 0.0

    async def get_zone_boost_temperature(self, zoneid):
        zone = await self.get_zone(zoneid)
        if zone is None:
            return 0.0
        return zone_boost_temperature(zone) or 0.0

    async def is_boost_active(self, zoneid):
        zone = await self.get_zone(zoneid)
        if zone is None:
            return False
        return zone_is_boost_active(zone)

    async def boost_hours(self, zoneid):
        zone = await self.get_zone(zoneid)
        if zone is None:
            return 0
        return zone_boost_hours(zone) or 0

    async def boost_timestamp(self, zoneid):
        zone = await self.get_zone(zoneid)
        if zone is None:
            return 0
        return datetime.datetime.fromtimestamp(zone_boost_timestamp(zone) or 0)

    async def is_target_temperature_reached(self, zoneid):
        zone = await self.get_zone(zoneid)
        if zone is None:
            return False
        current = zone_current_temperature(zone) or 0.0
        target = zone_target_temperature(zone) or 0.0
        return current >= target

    async def set_zone_target_temperature(self, zoneid, target_temperature):
        zone = await self.get_zone(zoneid)
        if zone is None:
            _LOGGER.error("Cannot set target temperature, unknown zone: %s", zoneid)
            return None
        return self._set_zone_target_temperature(zone, target_temperature)

    async def set_zone_boost_temperature(self, zoneid, target_temperature):
        zone = await self.get_zone(zoneid)
        if zone is None:
            _LOGGER.error("Cannot set boost temperature, unknown zone: %s", zoneid)
            return None
        return self._set_zone_boost_temperature(zone, target_temperature)

    async def set_zone_advance(self, zoneid, advance_state=True):
        zone = await self.get_zone(zoneid)
        if zone is None:
            _LOGGER.error("Cannot set advance state, unknown zone: %s", zoneid)
            return None
        return self._set_zone_advance(zone, advance_state)

    async def activate_zone_boost(self, zoneid, boost_temperature=None, num_hours=1, timestamp=0):
        zone = await self.get_zone(zoneid)
        if zone is None:
            _LOGGER.error("Cannot activate boost, unknown zone: %s", zoneid)
            return None
        return self._set_zone_boost(zone, boost_temperature, num_hours, timestamp=timestamp)

    async def deactivate_zone_boost(self, zoneid):
        zone = await self.get_zone(zoneid)
        if zone is None:
            _LOGGER.error("Cannot deactivate boost, unknown zone: %s", zoneid)
            return None
        return await self.activate_zone_boost(zoneid, boost_temperature=None, num_hours=0, timestamp=None)

    async def set_zone_mode(self, zoneid, mode):
        """
        Set zone mode; zoneid may be zone id or zone name.
        """
        assert isinstance(mode, ZoneMode)
        zone = await self.get_zone(zoneid)
        if zone is None:
            _LOGGER.error("Cannot set mode, unknown zone: %s", zoneid)
            return None
        modevalue = get_zone_mode_value(zone, mode)
        modeindex = GetPointIndex(zone, PointIndex.MODE)
        return self._set_zone_mode(zone, modevalue, modeindex)

    async def get_zone_mode(self, zoneid):
        zone = await self.get_zone(zoneid)
        if zone is None:
            return None
        return zone_mode(zone)

    def reset_login(self):
        self._login_data = None

    async def async_login(self):
        return await self._login()

    def __init__(self, username, password, cache_home=False):
        if cache_home:
            raise RuntimeError("cache_home not implemented")
        self._login_data = None
        self._user = {"user_id": None, "username": username, "password": password}
        self._homes = None
        self._home_details = None
        self.NextHomeUpdateDaytime = None
        self._refresh_token_validity_seconds = 1800
        self.http_api_base = "https://eu-https.topband-cloud.com/ember-back/"
        self.messenger = EphMessenger(self)
        # Note: Initial login must be triggered using async_login().
