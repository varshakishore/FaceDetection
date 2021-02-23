import requests
import argparse

parser = argparse.ArgumentParser(
    description="Image Background White-out"
)
parser.add_argument(
    "--directory", type=str, required=True, help="path to the input image"
)
parser.add_argument(
    "--input_name", type=str, required=True, help="the name of the input image"
)
args = parser.parse_args()

response = requests.post(
    'https://api.remove.bg/v1.0/removebg',
    files={'image_file': open("{}/{}.jpg".format(args.directory, args.input_name), 'rb')},
    data={'size': 'auto'},
    headers={'X-Api-Key': '3E5cFiKQfAyBz5zncGZ2fkph'},
)
if response.status_code == requests.codes.ok:
    with open("{}/{}_no-bg.jpg".format(args.directory, args.input_name), 'wb') as out:
        out.write(response.content)
else:
    print("Error:", response.status_code, response.text)