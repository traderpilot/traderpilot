#!/usr/bin/env python3
from traderpilot import __version__ as tp_version
from traderpilot_client import __version__ as client_version


def main():
    if tp_version != client_version:
        print(f"Versions do not match: \nft: {tp_version} \nclient: {client_version}")
        exit(1)
    print(f"Versions match: ft: {tp_version}, client: {client_version}")
    exit(0)


if __name__ == "__main__":
    main()
