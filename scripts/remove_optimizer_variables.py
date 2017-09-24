from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import re
import string

FLAGS = None
FILENAME_CHARS = string.ascii_letters + string.digits + '_'


def main():
  vars_dir = os.path.expanduser(FLAGS.output_dir)
  manifest_file = os.path.join(FLAGS.output_dir, 'manifest.json')
  with open(manifest_file) as f:
    manifest = json.load(f)
  new_manifest = {key: manifest[key] for key in manifest 
  if 'Adam' not in key and 'beta' not in key}
  with open(manifest_file, 'w') as f:
    json.dump(new_manifest, f, indent=2, sort_keys=True)

  for name in os.listdir(FLAGS.output_dir):
    if 'Adam' in name or 'beta' in name:
      os.remove(os.path.join(FLAGS.output_dir, name))



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--output_dir',
      type=str,
      required=True,
      help='The output directory where to store the converted weights')
  FLAGS, unparsed = parser.parse_known_args()
  if unparsed:
    print('Error, unrecognized flags:', unparsed)
    exit(-1)
  main()
