#!/usr/bin/env bash


curl -XPOST -H "Content-Type: application/json"  http://127.0.0.1:8000/infer -d '{"prompt":"# language: Python\n# write a bubble sort function\n"}'
