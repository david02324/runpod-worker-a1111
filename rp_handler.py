import time
import requests
import runpod
import threading
import redis
import json
import os
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.modules.rp_logger import RunPodLogger
from requests.adapters import HTTPAdapter, Retry
from schemas.api import API_SCHEMA
from schemas.img2img import IMG2IMG_SCHEMA
from schemas.txt2img import TXT2IMG_SCHEMA
from schemas.options import OPTIONS_SCHEMA

BASE_URL = 'http://127.0.0.1:3000'
REDIS_URL= os.environ.get('REDIS_URL')
TIMEOUT = 600

session = requests.Session()
retries = Retry(total=10, backoff_factor=0.1, status_forcelist=[502, 503, 504])
session.mount('http://', HTTPAdapter(max_retries=retries))
logger = RunPodLogger()
redis_client = redis.from_url(url=REDIS_URL)


# ---------------------------------------------------------------------------- #
#                              Automatic Functions                             #
# ---------------------------------------------------------------------------- #

def wait_for_service(url):
    retries = 0

    while True:
        try:
            requests.get(url)
            return
        except requests.exceptions.RequestException:
            retries += 1

            # Only log every 15 retries so the logs don't get spammed
            if retries % 15 == 0:
                logger.info('Service not ready yet. Retrying...')
        except Exception as err:
            logger.error(f'Error: {err}')

        time.sleep(0.2)

task_map = {}
def progress_T2I(event_id):
    previous_preview = ''
    while task_map[event_id]:
        time.sleep(1)
        res = requests.get(f'{BASE_URL}/sdapi/v1/progress')
        res_json = res.json()

        packet = {}
        packet['id'] = event_id
        data = {}
        if 'progress' in res_json and res_json['progress'] != None:
            data['progress'] = res_json['progress']
        if 'current_image' in res_json and res_json['current_image'] != None and previous_preview != res_json['current_image']:
            previous_preview = res_json['current_image']
            data['preview'] = 'data:image/png;base64,' + previous_preview

        packet['data'] = data

        channel = 'progress'
        message = json.dumps(packet)

        redis_client.publish(channel=channel, message=message)


def send_get_request(endpoint):
    return session.get(
        url=f'{BASE_URL}/{endpoint}',
        timeout=TIMEOUT
    )


def send_post_request(endpoint, payload, event_id):
    if endpoint == 'sdapi/v1/txt2img':
        non_blocking_thread = threading.Thread(target=progress_T2I, args=(event_id,))
        task_map[event_id] = True
        non_blocking_thread.start()

    res = session.post(
        url=f'{BASE_URL}/{endpoint}',
        json=payload,
        timeout=TIMEOUT
    )
    task_map[event_id] = False

    return res


def validate_api(event):
    if 'api' not in event['input']:
        return {
            'errors': '"api" is a required field in the "input" payload'
        }

    api = event['input']['api']

    if type(api) is not dict:
        return {
            'errors': '"api" must be a dictionary containing "method" and "endpoint"'
        }

    api['endpoint'] = api['endpoint'].lstrip('/')

    return validate(api, API_SCHEMA)


def validate_payload(event):
    method = event['input']['api']['method']
    endpoint = event['input']['api']['endpoint']
    subModels = []
    if 'subModels' in event['input']:
        subModels = event['input']['subModels']
    else:
        logger.log('No subModels')

    payload = event['input']['payload']
    validated_input = payload

    if endpoint == 'txt2img':
        validated_input = validate(payload, TXT2IMG_SCHEMA)
    elif endpoint == 'img2img':
        validated_input = validate(payload, IMG2IMG_SCHEMA)
    elif endpoint == 'options' and method == 'POST':
        validated_input = validate(payload, OPTIONS_SCHEMA)

    return endpoint, event['input']['api']['method'], validated_input, subModels


# ---------------------------------------------------------------------------- #
#                                RunPod Handler                                #
# ---------------------------------------------------------------------------- #
def handler(event):
    event_id = event['id']
    validated_api = validate_api(event)

    if 'errors' in validated_api:
        return {
            'error': validated_api['errors']
        }

    endpoint, method, validated_input, subModels = validate_payload(event)

    if 'errors' in validated_input:
        return {
            'error': validated_input['errors']
        }

    if 'validated_input' in validated_input:
        payload = validated_input['validated_input']
    else:
        payload = validated_input


    try:
        shouldRefreshLora = False
        for subModel in subModels:
            name = subModel['name']
            url = subModel['url']
            t = subModel['t']
            logger.log(f'Downloading {t}/{name} from: {url}')

            r = requests.get(url)

            if t == 'TI':
                p = f'/workspace/stable-diffusion-webui/embeddings/{name}'
                open(p, 'wb').write(r.content)
            elif t == 'LORA':
                shouldRefreshLora = True
                p = f'/workspace/stable-diffusion-webui/models/Lora/{name}'
                open(p, 'wb').write(r.content)

        if shouldRefreshLora:
            send_post_request('sdapi/v1/refresh-loras', {}, '')

    except Exception as e:
        return {
            'error': str(e)
        }

    try:
        logger.log(f'Sending {method} request to: /{endpoint}')

        if method == 'GET':
            response = send_get_request(endpoint)
        elif method == 'POST':
            response = send_post_request(endpoint, payload, event_id)
    except Exception as e:
        return {
            'error': str(e)
        }

    return response.json()


if __name__ == "__main__":
    wait_for_service(url='http://127.0.0.1:3000/sdapi/v1/sd-models')
    logger.log('Automatic1111 API is ready', 'INFO')
    logger.log('Starting RunPod Serverless...', 'INFO')
    runpod.serverless.start(
        {
            'handler': handler
        }
    )
