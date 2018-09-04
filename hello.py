#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-09-04 10:21:51
# @Author  : biofool (biofool@163.com)
# @Link    : https://github.com/biofoolgreen
# @Version : $Id$

from flask import Flask
app = Flask(__name__)


@app.route('/')
def hello_world():
    return "Hello, World!"
