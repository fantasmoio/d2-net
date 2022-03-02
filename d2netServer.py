#!/usr/bin/env python

import argparse

import d2net

# Argparse
parser = argparse.ArgumentParser(description = 'D2-net server script.')
parser.add_argument('--port', type = int, default = 8076, required=False, help='The port this server should listen on')
parser.add_argument('--method', default = 'torch', choices=['torch', 'trt', 'torchMulti'], required=False, help='The model type we would like to use')
parser.add_argument('--fp16', default = 'off', choices=['on', 'off'], required=False, help='Floating point 16 optimization')
args = parser.parse_args()

def main():
    method = args.method
    fp16 = args.fp16
    port = args.port

    d2Server = d2net.MainD2Thread(port, method, fp16)
    d2Server.start()

if __name__ == '__main__':
    main()
