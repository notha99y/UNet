import os
import json

unet_config = 'config/unet.json'
print('unet json: {}'.format(os.path.abspath(unet_config)))
config = json.load(unet_config)

print(config)
