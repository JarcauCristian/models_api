from redis import StrictRedis
from datetime import datetime, timedelta

redis_client = StrictRedis(host='192.168.1.2', port=6379, db=0)


def get_data_from_redis(key):
    return redis_client.get(key)


def set_data_in_redis(key, value, expire_time_seconds):
    redis_client.setex(key, expire_time_seconds, value)


def is_data_stale(key, expire_time_seconds):
    last_update_time_str = redis_client.get(f"{key}_timestamp")
    if last_update_time_str:
        last_update_time = datetime.fromisoformat(last_update_time_str.decode())
        current_time = datetime.utcnow()
        return (current_time - last_update_time) > timedelta(seconds=expire_time_seconds)
    return True


def update_timestamp(key):
    current_time = datetime.utcnow()
    redis_client.set(f"{key}_timestamp", current_time.isoformat())
